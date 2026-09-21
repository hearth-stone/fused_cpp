# 联合运行资源模型（2026-09-11，进行中）

> 当前任务状态以 [CURRENT.json](../CURRENT.json) 为准。本文为历史证据；其中设计和进度描述保留当时口径。

## 目标和边界

按用户授权推进完整目标：真实路径基线闭合、状态需求、共享资源与重叠、动态验证、
混合宽度、固定候选与相同预算 planner 搜索。当前不是已完成的模型，也不替换 planner 基线。
M 类 Lab 改动，生产 ABI/kernel/build/default 不变；回退边界为独立分析器和后续 Lab adapter。

固定当前 Arm-codex-internal、SVE BF16、H4096/F512，先8T再1/2/4/8/16T实际宽度组合。
early merge关闭，目标为所有expert计算完成，gather和间隙纳入，最终merge单列。
真实窗口来自计划，不将窗口重新扩成搜索变量；8T当前full stripes为(8,0,0,1,1)，
Ntile16、完整W13/W2权重8/4MiB、owner1MiB/512KiB。

## 冻结的验收目标

- 基础成本主要分组有符号偏差±3%，短时间点同时报告测量噪声。
- 未见联合计划完成MAPE<=3%，P90相对绝对误差<=5%；M/宽度/尾块/背景分别报告。
- 同一实测候选集合选择损失median<=2%、P90<=5%，噪声内视为近似并列。
- 固定集合评分后，再做相同搜索预算完整对照，报告评分/搜索开销。
- 留出按完整M/历史/混合比例/顺序/team组合划分；开发见过的数据不得重新称为独立验收。
- 第一阶段重复会话只验证可重复性，不算未见M或联合泛化。

## 执行顺序

1. 已完成8T范围：真实路径分解、上下文对照和同权重嵌套前缀辨识；其他宽度仍需验证。
2. 8T/M<=60的六类GEMM历史候选已完成18个前瞻点验证；gather/初始入口候选尚未完整联合验收。
3. 下一步：真实panel观测及状态相关流量/请求量，区分LLC与DRAM服务、驻留与流式、同域与跨LLC对照。
4. 待做：资源服务容量约束、减速后的请求反馈、计算访存重叠；仅拟合可辨识参数。
5. 待做：实测时间线条件诊断与自主动态模拟对照；扩展混合宽度和新鲜留出。
6. 待做：真实trace、固定候选集合与相同预算planner搜索，满足验收才考虑替换基线。

## 首项实现与历史证据复核

`analyze_joint_baseline.py`读取已验证的真实路径compact trace，逐样本核对阶段端点、
duration、顺序、非有限数及总和；分别保存gather、W13、W2、各阶段前后间隙。
使用均值作可加分解；各项中位数不保证可加，不能用中位数相加冒充总中位数。
检查frontier、seed、runner/extension/workspace身份及31个非warmup样本。

历史24个形状/输入条件，每条件两会话31样本：session1各项中位数预测session2，
gather MAE3.7475us/MAPE7.2872%，W13 1.3225us/0.3914%，W2 0.7550us/0.4722%，
间隙0.09625us/3.3705%，总时间4.9179us/1.0353%。这不是新模型泛化结果。
真实路径的GEMM重复性好；不能依据另一协议的连续链偏差直接修改真实路径成本。

现有连续链使用32份常数权重/输入、64阶段lead-in和8expert批次；真实路径使用实际route、
四份权重、scrub和固定workspace。两者状态不等价，时间差目前不作单一机制因果归因。

## 新会话采集

使用已验证runner，未改native计时或kernel。NUMA3、CPU240–319，目标8T312–319，
其余expert由依赖等待目标完成；median/uniformish各12个目标，附anchor，5warmup/31样本。
新seeds611081/611082，复用旧frontier与输入，输出只写新目录。
预先确认runner SHA ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba，
扩展SHA dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f，
profile SHA cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41。
普通页沿用原协议，无新增HugeTLB策略。源码身份与历史一致。
四个新会话均完成，2084次调用通过数值/isolation/trace校验。

| 指标 | 新session1预测新session2 MAE/us | MAPE/% | 历史session1预测两个新session MAE/us | MAPE/% |
|---|---:|---:|---:|---:|
| gather | 4.4150 | 8.6705 | 5.7513 | 11.3591 |
| W13 | 1.3625 | 0.4460 | 1.5971 | 0.5141 |
| W2 | 0.8046 | 0.4732 | 0.7069 | 0.4200 |
| 间隙 | 0.4617 | 14.3725 | 0.2554 | 8.2944 |
| 完整expert | 4.9833 | 1.0471 | 5.7954 | 1.1632 |

跨日比较48个条件会话，仍为同形状lookup而非泛化模型。
间隙相对误差大但绝对值小；gather波动不能作为GEMM响应修正。
M48真实路径W13的历史预测减新实测平均仅+0.355us，W2为-2.25us；
M13 W13为-1.065us、W2为+0.270us。此前连续链的M48基线低估/M13高估
结论仅适用于那套协议，不能据此认定真实路径T0有同方向大偏差。
完整证据见`fresh_audit.json`；`baseline_audit_checked.json`增加frontier身份审计，
保留早期`baseline_audit.json`，数值不改。模型尚未解释跨协议差异的具体机制。

本地/远端产物根`tmp/joint_cost_model_20260911`，远端项目根`/home/zhangxu/codex/fused_cpp`。
历史输入根`tmp/eight_single_expert_20260910`；新采集shell为`run_baseline.sh`。

```sh
.venv/bin/pytest -q tests/test_moe_joint_baseline.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_joint_baseline.py \
  --root tmp/eight_single_expert_20260910 \
  --output tmp/joint_cost_model_20260911/baseline_audit.json
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_joint_baseline.py \
  --root tmp/joint_cost_model_20260911 \
  --reference tmp/joint_cost_model_20260911/baseline_audit_checked.json \
  --output tmp/joint_cost_model_20260911/fresh_audit.json
# 远端，已有输出时不可重复覆盖
bash tmp/joint_cost_model_20260911/run_baseline.sh
```

## 下一批真实形状网格

`prepare_joint_baseline.py`复用真实route isolation helper，已生成而未采集：
训练M1–24及36/48/60（27条件），验证M25/26/27/29/30/31/32/35/39/40/41/42/43/47（14条件）。
训练包含全部小行首块、h1尾块和多个完整块数；验证将后续尾块整体排除在拟合外。
部分验证M有历史实验记录，本轮称为拟合留出，不称为从未观察过的形状。
缺失M28/33/34/37/38/44/45/46，后续需其他真实route补齐，不能声明全部尾块覆盖。
网格原始输入/计划身份、M分组和目标核位置保存在`shape_grid/protocol.json`及train/validation.json。
验证集须在拟合参数冻结后采集，训练集可先采。

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_joint_baseline.py \
  --source tmp/planner_repaired_pressure_20260910 \
  --output tmp/joint_cost_model_20260911/shape_grid
```

首次`shape_grid/run_train.sh`在采集前被旧runner拒绝：`max-plans must be in 1..13`，
没有产生训练测量。保留原脚本和stderr，没有改runner或放宽上限。
准备器新增分批：每批最多12目标+anchor，检查所有目标恰好出现一次。
修正运行位于`shape_grid_batched/run_train.sh`，seeds611101/611102、train_01/02/03，
每批每会话5warmup/31轮，两会话，已顺序启动；验证集仍未启动。
整组27/14条件和原始plan身份保持不变；分批会话状态需用anchor和重复性检查。

当前14个定向测试通过（新分解/分组/历史预测/分批8个、原isolation6个）；Ruff及diff检查通过。
完整资源模型、未见条件验证与planner效果均未完成。

## 基础成本组合候选（已冻结，留出进行中）

新增`joint_baseline_model.py`，预先固定主假设为后续尾块增量不变，诊断对照为
按同历史完整块增量缩放尾块。两者均只使用训练session1，session2只作重复性验证。
基础成本限定8T、M1..60、当前真实路径协议，不能外推到其他宽度/窗口或更大M。

令`P_s(b)=T_s(12b)`来自完整M训练点，`D_s(r)=T_s(12+r)-T_s(12)`来自h1尾块。
主预测为`T_s(12b+r)=P_s(b)+D_s(r)`；r=0直接返回完整前缀，M<=12保留逐行首块成本。
诊断版本将`D_s(r)`乘以`[P_s(b+1)-P_s(b)]/[P_s(2)-P_s(1)]`。
这是待验证的跨历史迁移假设，不是新测得的逐panel服务量；若增量非正会拒绝拟合，
不截断数据掩盖不可辨识性。阶段启动已包含在前缀内，不再重复添加。
gather独立拟合非负`g0+g1*M`，间隙使用训练条件中位数；报告两项各自误差，不将它们
吸收到GEMM参数。此简单gather响应仍需验证，不能默认满足泛化门槛。

18个新增模型测试通过，覆盖首块/完整块/未见尾块组合、重复启动避免、拟合不读取
session2、留出M不能进入训练、非正增量拒绝及域外拒绝。
训练参数、两种版本14个留出预测应在验证采集前落盘，主版本不按验证分数事后更换。

六个训练会话已全部完成，2406次调用通过验证。session1拟合、session2验证的主版本：
W13 MAE1.3211us/MAPE0.6787%，W2 0.5933us/0.6728%，完整expert 4.7775us/1.4642%，
完整expert P90相对误差3.8601%。这仍是同形状重复验证，两种尾块假设在训练M集合上相同。
gather包络仿射参数为46.7645+0.32574*M us，间隙2.92us；不用于任何资源需求计算。

`shape_grid_batched/model.json`保留原冻结候选，`model_scoped.json`补充明确的
`isolated_first_expert`执行域及gather解释，参数和全部留出预测逐字段相同。
后者SHA998ff39bae388da1a80a3e70d9568206727ad7a43b3df2485610906c84124b01，
已在`tail_holdout/model_before_measurement.json`保存测前副本。
14个留出条件分两批，seeds611121/611122，每批5warmup/31轮、两会话，
`tail_holdout/run_holdout.sh`已经启动，结果尚未完成，不能声称尾块迁移通过。

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_joint_baseline.py \
  --root tmp/joint_cost_model_20260911/shape_grid_batched \
  --output tmp/joint_cost_model_20260911/shape_grid_batched/audit.json
.venv/bin/python optimizations/fused_moe_sve/benchmarks/joint_baseline_model.py fit \
  --audit tmp/joint_cost_model_20260911/shape_grid_batched/audit.json \
  --output tmp/joint_cost_model_20260911/shape_grid_batched/model_scoped.json
# 远端，测前模型与脚本已保存
bash tmp/joint_cost_model_20260911/tail_holdout/run_holdout.sh
```

## Gather包络的到达错位（已验证诊断）

只读原始worker trace，6个训练会话均验证原trace SHA和完整call/phase数。
脚本`tmp/joint_cost_model_20260911/gather_worker_diagnostic.py`在采集结束后运行，
没有新增计时或benchmark，输出`gather_workers*.json`，每M每会话31个样本。
对每个样本定义`E=max(end_i)-min(start_i)`；令j为最后结束worker，则精确有
`E=[start_j-min(start_i)]+[end_j-start_j]`。第二项是worker计时间隔，并非纯访存服务。

| M | gather包络/us | 最大worker区间/us | 最后结束worker到达延迟/us |
|---|---:|---:|---:|
| 1 | 45.790 | 1.870 | 44.375 |
| 8 | 52.310 | 3.510 | 50.440 |
| 12 | 56.530 | 5.395 | 52.985 |
| 13 | 45.450 | 9.050 | 37.875 |
| 24 | 50.210 | 11.950 | 40.730 |
| 48 | 66.280 | 23.545 | 44.665 |
| 60 | 68.985 | 30.965 | 43.360 |

表为两会话各自中位数的均值，不具有逐列可加性。小M包络主要由worker到达错位组成，
并非gather实际搬运几十us。源码trace结束位于gather后的barrier之前；包络混入的是
跨worker启动时差，不能统一归为barrier内部等待，更不能全部算作LLC/DRAM需求。
具体启动时差机制尚未辨识。下一步资源模型需要显式worker/team到达状态，并区分
队列首expert与后续expert；当前gather仿射只作为相同隔离首任务协议的完成时间对照。

最新32个定向测试通过，Ruff检查通过；未改生产默认或完整planner。

## 后续尾块留出结果（完成，未通过分阶段验收）

`tail_holdout`四会话1284次调用全部通过数值与trace校验；14个M、两会话共28条件会话，
使用测前SHA998ff39b…的模型，逐字段复核保存的预测与当前公式相同，无重拟合。

| 指标 | 主版本MAE/us | 主版本MAPE/% | 完整块缩放对照MAE/us | 对照MAPE/% |
|---|---:|---:|---:|---:|
| W13 | 5.3189 | 1.0224 | 7.0703 | 1.3831 |
| W2 | 13.9243 | 5.8632 | 10.7548 | 4.5071 |
| gather包络 | 3.6204 | 6.0258 | 3.6204 | 6.0258 |
| 完整expert | 15.0696 | 1.8896 | 14.2393 | 1.7867 |

主版本完整expert P90误差3.9385%，但W2所有条件高估、有符号均值+13.9243us，
不满足基础阶段分组±3%的目标。对照W2仍有+10.6700us偏差，不能凭总时间较准采用。
M25/26/27 W2高估约12us，M31/32约17–19us，M43约21.58us；M35/47的11行尾块
只约4.81/3.60us，说明偏差依赖尾块类别和历史，不能仅统一缩放完整块响应。
这些是整阶段跨M组合残差，尚不能认定全部发生于尾块内部，需进一步辨识完整前缀/尾块。
结果在`tail_holdout/audit.json`、`evaluation.json`，两个候选均保留，未切换基线。

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_joint_baseline.py \
  --root tmp/joint_cost_model_20260911/tail_holdout \
  --output tmp/joint_cost_model_20260911/tail_holdout/audit.json
.venv/bin/python optimizations/fused_moe_sve/benchmarks/joint_baseline_model.py evaluate \
  --audit tmp/joint_cost_model_20260911/tail_holdout/audit.json \
  --model tmp/joint_cost_model_20260911/tail_holdout/model_before_measurement.json \
  --output tmp/joint_cost_model_20260911/tail_holdout/evaluation.json
```

## 同核首任务/后续任务对照（完成）

新增`joint_worker_timing.py`保留各worker相对scheduled_compute起点的时间，按实际trace
标记该任务是否各worker的首任务；这个实测标签只作诊断，不可作为预测时的oracle输入。
`analyze_workspace_isolated_width.py --worker-details`为可选输出，默认旧格式保持不变。
在旧train_03/session1的161次call上验证，去除新worker字段后与旧compact逐字段完全相同。

旧trace中M48：隔离首任务gather包络69.71us、最大worker23.90us；anchor中为24.57/23.53us，
后者为队列后续任务。M60分别69.46/31.60和28.29/27.78us。
这不是同核无竞争对照，anchor有竞争且放置不同，只支持继续独立测量，不能据此归因全部差异。

`prepare_joint_context.py`使用同一真实target、同8T/CPU312–319，比较首任务与先执行一个
真实M36 leader后的任务；六个target M1/8/12/13/24/48，共12计划+anchor。
所有其他expert等待target结束，leader与target串行共用8核，原其他依赖保留。
分析器仅对显式`isolated_predecessors`允许先行任务，并核对其结束先于target开始；
任何其他重叠仍拒绝。未修改runtime、kernel或原bounded runner。
这会同时改变到达和cache状态，不称为只改变唤醒的单因素实验。

计划和协议`context/context.json`、`protocol.json`，seeds611141/611142、同4copies/
216MiB scrub/5warmup/31轮，`context/run_context.sh`两轮已完成，共1042次调用校验通过。
新分析器在`worker_runner/`隔离快照内，后处理增加worker输出，不给native增加计时。
后续用`analyze_joint_context.py`比较同pair中位差和各阶段worker区间，不把差异压进统一倍率。

最新53个定向测试通过，Ruff通过。总代价模型、资源响应及planner验收仍未完成。

`context/analysis.json`保存所有样本和两轮独立统计。下表是每轮同pair后续减首任务
差值中位数的两轮均值，单位us；独立中位数不保证逐列可加。

| M | 完整expert变化 | gather包络变化 | gather最大worker变化 | gather到达跨度变化 | W13变化 | W2变化 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | -43.900 | -41.630 | +0.710 | -42.440 | -1.620 | -1.200 |
| 8 | -40.455 | -38.335 | +1.050 | -39.780 | -1.300 | -0.260 |
| 12 | -38.780 | -34.590 | +0.455 | -35.965 | -0.585 | +0.740 |
| 13 | -43.900 | -37.255 | +0.865 | -40.315 | -3.155 | -0.965 |
| 24 | -39.425 | -37.960 | -0.355 | -39.685 | -3.580 | -0.355 |
| 48 | -47.050 | -35.410 | -1.680 | -37.950 | -5.940 | -1.615 |

间隙变化约-1.53至-1.67us。大幅变化主要在worker到达，而非gather自身worker区间。
W2上下文变化远小于尾块留出的平均13.92us偏差，故启动错位不能解释全部W2历史残差。
下一步将初始worker可用时间与每任务gather区间独立建模；不能给每个expert重复添加
隔离首任务的约40us到达跨度。W2前缀/尾块残差继续独立处理，不能拿启动修正抵消它。
当前跨M残差还混合了不同expert的权重/route布局，需要同权重及嵌套route前缀对照
或真实路径panel级证据来辨识，不直接把所有阶段差值称为纯尾块成本。

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_joint_context.py \
  --source tmp/planner_repaired_pressure_20260910 \
  --output tmp/joint_cost_model_20260911/context/context.json
# 远端，使用独立worker_runner分析器快照
bash tmp/joint_cost_model_20260911/context/run_context.sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_joint_context.py \
  --root tmp/joint_cost_model_20260911/context \
  --output tmp/joint_cost_model_20260911/context/analysis.json
```

## 同权重、同route前缀的M对照（完成）

`joint_nested_routes.py`保持target expert13（原M60）相同packed weights、hidden和保留route的
输入/输出位置。较小M仅将该expert后缀route转给已有、该token尚未选择的expert，保持top-k
唯一性、总route数量及active expert集合。其他expert依赖target结束，不产生同时竞争。
scratch分配状态仍可能随M变化，因此这是减少混杂的整阶段对照，不是直接panel计时。

`bench_bounded_order_extension.py --nested-routes`只为此Lab模式启用变体输入；每种输入
分别用原anchor得到参考，再检查4份权重、poisoned workspace，不能共享不同输入的参考值。
原模式保持原检查次序与执行输入。新增trace_call_prefix记录实际正确性调用序列；
解析器检查序列及每次测量的W13/W2实际rows与声明M一致。新控制不进入生产API/默认build。

row1批次M1/12/13/24/25/36/37/48/49/60，rows7_8_11批次M12/19/20/23/24/31/32/35/36/43/47/48。
每批单独进程内共享B及输入前缀，不将跨批次差分称为同一物理分配；seeds611161/611162。
先通过row1正确性烟测，再运行两批两会话5/31，1968次调用均通过数值/隔离/实测M/trace校验。
native扩展与workspace不变；独立nested_runner快照SHA5da239667ce370ced464100fd7c6f1ac4db097cd0c96c8f8d1ea0ad56a72e6d5。

下表为每会话同pair `T(12h+r)-T(12h)` 的中位数，再取两会话均值，单位us。

| 行数r | 历史h | W13净增量 | W2净增量 |
|---|---:|---:|---:|
| 1 | 1 | 105.365 | 49.085 |
| 1 | 2 | 98.410 | 38.210 |
| 1 | 3 | 96.980 | 29.515 |
| 1 | 4 | 96.255 | 21.515 |
| 7 | 1 | 127.960 | 60.940 |
| 7 | 2 | 116.265 | 43.495 |
| 7 | 3 | 111.165 | 40.020 |
| 8 | 1 | 127.810 | 61.990 |
| 8 | 2 | 115.270 | 43.510 |
| 11 | 1 | 156.255 | 83.455 |
| 11 | 2 | 154.980 | 80.330 |
| 11 | 3 | 153.390 | 79.525 |

控制权重和route前缀后，历史依赖仍清晰存在，不能仅归因于之前换expert。
row1第一轮各worker自己的W2整阶段差分也由h1约48–51us降到h4约23–26us，
不只是max-worker聚合造成的差异；但仍未定位到同一call内哪一个panel。
结论支持按kernel类别拟合历史响应，不支持所有尾块统一倍率或全都照搬h1。
完整数据及配对差分在`nested/analysis.json`；这是机制训练数据，不当作新模型独立验收。

`nested_kernels`已准备并启动补采：同expert下M1–12首块，以及3/5/9行代表kernel的h1/2/3；
后者M12/15/17/21/24/27/29/33/36/39/41/45。seeds611181/611182，仍每批<=13计划、两会话5/31。
与已测1/7/11行合起来覆盖六个计算行类别，结果未完成，参数未据此拟合。

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_joint_nested.py \
  --source tmp/planner_repaired_pressure_20260910 --output tmp/joint_cost_model_20260911/nested
# 远端
bash tmp/joint_cost_model_20260911/nested/run_nested.sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_joint_nested.py \
  --root tmp/joint_cost_model_20260911/nested --output tmp/joint_cost_model_20260911/nested/analysis.json
```

## 初始worker入口时序项（候选，尚未接入完整预测器）

`joint_worker_ready_model.py`显式使用`start_i=max(R_task,A_i)`、`finish=max_i(start_i+G_i)`。
R_task是依赖完成时间，A_i是该worker本次call的初始入口代理，G_i为单独提供的工作区间。
后续任务的R_task覆盖已过去的A_i，因此不会重复收取启动跨度；模型不把A_i算成资源需求。
初始入口包含dispatch，不声称纯硬件唤醒。当前还没有独立预测全部G_i或接入planner。

可选worker解析新增dependency-free任务入口，原始记录保持；只用anchor全80核覆盖样本校准。
在`initial_ready`用context/session1训练，context/session2重复及uniformish两会话迁移诊断。
4份旧trace重解析全部通过校验，没有重跑kernel。解析首次因缺少uniformish frontier副本退出，
补齐同SHA副本后仅恢复剩余3份，保留原脚本和已完成context/session1，未重启测量。

| 入口指标 | context第2轮 | uniformish第1轮 | uniformish第2轮 |
|---|---:|---:|---:|
| 单次worker入口MAE/us | 24.067 | 23.548 | 22.753 |
| 单次worker入口P90绝对误差/us | 48.100 | 50.570 | 44.160 |
| 各CPU跨31次中位入口MAE/us | 8.010 | 4.908 | 5.895 |
| 全池最后入口中位偏差/us | +6.900 | +0.520 | +2.350 |
| 全池最早入口中位偏差/us | +34.870 | +30.960 | +32.600 |

固定每CPU中位表不能还原每次入口顺序/分散程度，尤其最早入口存在统计聚合偏差。
这些是回顾性入口诊断，不是完整gather或联合完成时间精度；保留其不确定性，尚不采用。
若后续联合预测需要，必须保留入口之间的相关性或验证按team聚合的近似，而非拟合每次噪声。
`calibration_with_summary.json`记录完整指标，早期远端`calibration.json`保留同参数的较少指标版本。

新增函数的不变量及相关旧路径共82个定向测试通过；Ruff和diff检查通过。
完整目标仍包括补齐基础历史曲线、状态需求、共享服务反馈、动态/混合宽度及planner验收。

## 六类kernel历史模型（18点前瞻验证完成）

`nested_kernels`两批两会话2132次调用校验完成。`joint_kernel_history_model.py`仅取
`nested`及`nested_kernels`的session1奇数行代表r1/3/5/7/9/11在h1/2/3的配对净增量，
每个GEMM独立拟合6条`D_g(h)=a_g+b_g*exp(-lambda_g*(h-1))`，a/b/lambda非负。
这是限定h1..4的经验函数，a不是独立测得的计算下限，不用于推断物理计算占比。
完整前缀P(12b)取训练完整M点的中位数；首块M1..12来自同权重首块数据。
偶数尾行r共享r-1的计算行类别，增加首块差值`F0(r)-F0(r-1)`作为待验证的偏移。
整体`T(12b+r)=P(12b)+D_g(b)+offset_r`，r=0仅取完整前缀，不重复加入stage setup。

模型仅输出W13/W2，域为8T、H4096/F512、Ntile16、(8,0,0,1,1)、M1..60；未含gather或初始入口。
所有预测在采集前冻结，model SHA50923778751e70fe9b630f24ac0f14fcac5239b53946cd335938a55e8e159e70。
18个新点为M14/16/18/22/26/28/30/34/38/40/42/44/46/50/52/54/56/58；
seeds611201/611202，两批两会话1640次调用均通过数值/隔离/实测M/trace校验。
`history_validation/model_before_measurement.json`、freeze_record.json及evaluation.json保留冻结链。

| 新鲜验证指标（36条件会话） | MAE/us | MAPE/% | P90相对绝对误差/% |
|---|---:|---:|---:|
| W13 | 2.9689 | 0.6593 | 1.3631 |
| W2 | 1.3839 | 0.5252 | 1.1121 |
| 双GEMM | 3.0554 | 0.4581 | 0.9822 |

W13最大误差10.2533us（M22/session1，约3.33%），W2最大7.6684us（M56/session1，约2.16%），
并非每个单点都小于3%。按h1..4的阶段分组误差在当前门槛内；这里只通过本次8T基础GEMM
检查，不证明联合运行、其他宽度/窗口或更大M已经可靠。
此前已查看的真实route14点回顾性迁移，W13/W2 MAPE0.3904%/0.4741%、MAE1.8408/1.1445us；
不将这份历史数据再次称作前瞻留出。旧常量尾块版本及其W2高估结果继续保留。

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/joint_kernel_history_model.py \
  --analysis tmp/joint_cost_model_20260911/nested/analysis.json tmp/joint_cost_model_20260911/nested_kernels/analysis.json \
  --output tmp/joint_cost_model_20260911/kernel_history/model.json
# 远端，18点预测先冻结
bash tmp/joint_cost_model_20260911/history_validation/run_validation.sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/evaluate_joint_kernel_history.py \
  --model tmp/joint_cost_model_20260911/history_validation/model_before_measurement.json \
  --analysis tmp/joint_cost_model_20260911/history_validation/analysis.json \
  --output tmp/joint_cost_model_20260911/history_validation/evaluation.json
```

## Gather工作量与worker区间候选

`joint_gather_work.py`镜像实际M-panel/K-stripe整数映射，明确9–11行打包到12行，
K边界按8元素对齐，保留分片数量/读入逻辑字节/填充后写字节/输入64B-line覆盖。
使用已有native测试API，1/2/4/8/16T × M1..60共300配置逐tuple完全一致；并非这些宽度的
时间模型都已校准。首次查询缺少src导入路径，补齐后只执行整数映射，未运行性能kernel。
字节与line覆盖是算法工作量，不等同实测LLC/DRAM流量。

`joint_gather_model.py`在两组nested训练session1的worker区间上拟合非负线性系数，
特征为constant/read_KiB/write_KiB/segments，系数[0,0.342591,0.075170,0]，设计矩阵条件数194.57。
仅在8T/4096下拟合，没有把到达跨度当作gather工作；系数不是独立硬件带宽。
训练368个worker条件MAE1.2926us，第二会话1.2108us；未用于拟合的18点两会话288个worker
条件MAE1.8133us、MAPE15.9391%、bias+0.5224us、max8.6823us。
这一gather检查在数据采集后进行，是回顾性形状检查，精度仍弱于GEMM；不能据此声称完整
expert或联合模型通过。源数据、几何oracle及gather_model.json均保留。

实机cache几何另由sysfs核实：CPU312的L2为1280KiB/10-way/64B line，LLC为71680KiB/
28-way/128B line、共享CPU280–319；L1d为64KiB/4-way/64B。容量不等同服务带宽，后续单独测量。
只读ELF检查确认当前`jit::get_kernel`为导出符号且经PLT调用，可评估Lab包装进行真实panel观测；
尚未构建/加载观测库，不修改生产JIT。优先获取真实panel区间和资源事件，避免仅凭整阶段差分
推断内部执行位置，再接共享服务、动态与混合宽度验证。

新增历史/工作量/拟合测试及相关旧路径共153个测试通过；native工作映射300点通过，Ruff通过。
生产kernel、默认planner与原fallback对照均保持不变，完整目标仍未完成。

## 真实panel观测实现（进行中）

新增独立Linux/AArch64共享库`joint_panel_observer.cpp`，以进程级LD_PRELOAD包装已导出的
`jit::get_kernel`，调用显式验证的原DSO实现，再对选定expert的4份W13/W2 B指针进行过滤。
生产源码、JIT机器码、全局加载配置和默认构建不变。仅支持本轮BF16 W13/degree5和W2Direct。
每个pinned CPU使用预分配、128B对齐的缓冲区；记录kernel前后CLOCK_MONOTONIC、CPU、
M/K/N、n_begin、A/B地址。没有panel barrier，kernel内部没有分配/锁/文件I/O。
计时与CPU查询是显式观测开销，算子返回并同步worker之后才写日志；必须验证其扰动。
新C ABI由串行Python controller调用，禁止并发native调用或重新配置；原DSO保持加载以维持指针生命周期。

`joint_panel_observer.py`保持packed tensors存活，标记每个call及control/capture状态。
同进程M12/13/24/25/48/49各有control/observe一对，共享route tensor和parsed plan对象，
每个输入仍分别对照anchor检查4份权重与poisoned workspace。Control不包装/计时target调用，
两者都保留getter的轻量转发；native阶段计时不包含返回后日志落盘。
预设扰动门槛：主要M/stage的同pair中位变化绝对值<=1%，同时报告us、噪声及异常点。
该门槛未通过前，不用panel数据替代已验证基础成本或拟合资源响应。

`analyze_joint_panels.py`验证完整call序列、control零记录、每CPU/每stage的12块/尾块序列、
K/N/n_begin、B copy指针、A前缀推进和W13→W2 barrier；不把不同worker的panel包络简单相加。
观测二进制在`tmp/joint_cost_model_20260911/panel_observer/observer.so`，构建用O3/C++17/
Wall/Wextra/Werror/fPIC/shared，无生产扩展重编译。首次缺少命名空间闭合导致编译失败，
修正后构建成功，尚未形成性能结论。
源码/构建身份、frontier/protocol在panel_observer目录，runner在独立panel_observer_runner快照。
当前正确性烟测进行中；资源PMU事件尚未加入，当前只记录真实kernel区间。


## Panel烟测校验与配对扰动验证启动

烟测已结束，`analyze_joint_panels.py`离线确认65次调用、1088条panel记录，
包含13个计划各自的anchor和四份权重正确性检查；control零记录、块序列、CPU/几何、
A/B地址和阶段先后约束均通过。`smoke_validated.json`保存输入身份与校验数量。
本地命令`.venv/bin/pytest -q tests/test_moe_joint_panels.py tests/test_moe_bounded_order_extension.py`
得到23 passed；这只证明当前观测/校验路径，不证明观测扰动或联合预测精度。

正式扰动脚本为`tmp/joint_cost_model_20260911/panel_observer/run_paired.sh`，
沿用NUMA3/CPU240–319、目标312–319、8T full stripes、5warmup/31轮和4份权重。
两场seed611221/611222顺序运行，13计划包含六个M的control/observe及anchor，
各调用同时输出原native阶段trace及可选panel记录。独立校验器在CPU0运行，先完成
上一场后处理再开始下一场，不与下一场性能采集并行。所有输出采用新路径，未覆盖烟测。
当前状态为正式测量进行中；预设1%门槛及原planner基线均不变。


## Panel配对扰动验证完成：当前观测版不通过采用门槛

`run_paired.sh`两场均正常结束；每场533次native调用、10880条panel记录通过数值、
完整call顺序、isolation、真实M/CPU/地址/几何及阶段顺序校验，总计1066次调用、21760条记录。
测量配置及seed同上一节，正式产物为`panel_observer/paired_session{1,2}*`；
原始trace和panel JSONL保留在Arm-codex-internal同路径，紧凑结果本地已取回。
`paired_runtime_identity.txt`保存本次runner/observer/profile/extension身份。

配对统计严格按同session/pair/copy比较observe与control；下表为每组31对相对差的中位数，
不是两组中位数的比值。`evaluate_paired.py`生成`perturbation.json`，保留绝对us、MAD及最大单对偏差。

| M | W13 session1/% | W13 session2/% | W2 session1/% | W2 session2/% |
|---|---:|---:|---:|---:|
| 12 | +0.764 | +1.002 | +1.114 | +1.507 |
| 13 | +0.143 | +0.343 | +1.105 | +1.132 |
| 24 | +0.747 | +0.727 | +0.435 | +1.480 |
| 25 | +0.719 | +0.614 | +1.270 | +1.491 |
| 48 | +0.559 | +0.762 | +0.390 | +0.869 |
| 49 | +0.982 | +0.348 | +1.949 | +2.079 |

W2/M49配对中位绝对差+6.65/+7.15us，M12为+0.92/+1.23us，M13为+1.45/+1.48us，
M25为+2.55/+3.02us。gather配对变化范围-14.232%..+12.717%，多组跨会话变号；
它没有被panel直接计时，不能把这些变化全部归为kernel包装开销，也不能用其抵消W2扰动。
预设1%门槛未通过，拒绝将当前panel观测作为无扰动成本或资源拟合输入；不放宽门槛、不改planner。

仅作为后续定位线索，观察到W2的1行尾块worker-panel中位数两场分别为：
h1(M13)49.810/49.890us，h2(M25)44.375/45.000us，h4(M49)38.940/39.485us。
M49的前四块也较M48的对应块短；因此整段净增量不能直接等同物理尾块。
这些量仍受未通过门槛的观测方式影响，不给出无扰动因果结论。
`decompose_panels.py`生成`panel_decomposition.json`，用最后完成worker的prefix/tail/service、
初始偏移和内部gap精确闭合每call的panel包络；均值可加，中位数不可强行相加。
下一步降低计时与记录扰动并重新做独立配对验证，再复核prefix变化和真实tail成本；
基础历史候选保持原状，资源容量/请求反馈/动态混合宽度/planner目标仍未完成。


## 观测重复发布消融（进行中）

原getter在每个worker每次解析时都向共享`g_kernels[stage][rows]`执行release store，即使
原JIT返回的指针没有变化。Lab变体仅增加acquire load比较，相同时不写；首次或指针
改变时仍release发布，wrapper仍acquire读取。配置、原getter调用、CPU查询、时钟、
buffer和日志格式均不变，没有新增生产开关或依赖。这是待证实的共享写开销假设。

新目录`tmp/joint_cost_model_20260911/panel_observer_cached`保留源代码、build脚本、
protocol及相同frontier；原panel_observer目录和二进制保留。seeds611241/611242，
两场control/observe同进程配对，5warmup/31轮、NUMA3及8T几何同上。旧runner快照复用。
目标原生构建通过Wall/Wextra/Werror，源码差分仅上述缓存发布判断；clang-format通过。
`run_smoke.sh`成功后才执行`run_paired.sh`，当前仍在验证中，不能声称扰动已经下降。


补充原观测数据的跨M地址审计：两会话×两stage×8CPU共32组中，M12/13/24/25/48/49
各自的首panel A虚拟基址均完全相同（每组所有正式样本只有同一个地址）。
`panel_observer/cross_m_a_pointer_audit.json`保存逐组结果。这排除了本组前缀时间变化仅由
A基址随M切换造成的解释；不证明相同缓存初态、物理页或访存路径。新缓存发布变体
烟测已通过65次调用/1088条panel完整校验，正式两场已进入运行。


缓存发布变体两场已完成，每场533调用/10880panel通过数值及trace完整性校验。
但1%门槛仍未通过：W2/M12为+1.622%/+1.004%，M25为+1.686%/+1.231%，
M49为+0.653%/+1.519%；W13/M12也为+1.039%/+1.151%。不能依据第一场M49下降
声称共享写是主因；旧/新库没有在同一个进程中直接配对，只能分别比较各自control/observe。
不采用当前观测时间校准成本，原子写消融保留为有界Lab对照，详细结果在
`panel_observer_cached/perturbation.json`。
下一诊断为相同目标CPU上的clock_gettime和sched_getcpu暖态调用开销，独立短程序
`panel_observer_cached/clock_cost.cpp`；不把暖态微测时间直接从真实kernel测量中扣除。


CPU312/NUMA3暖态调用诊断完成，g++ C++17/O3/Wall/Wextra/Werror，31轮每种200000次，
3预热轮，四种循环轮换顺序；CPU校验通过。单clock_gettime中位23.4096ns、sched_getcpu
111.4420ns、一次CPU查询加两次时钟161.3077ns（各轮范围161.0911..162.0488ns）。
原始`clock_cost.jsonl`及源码/二进制留在变体目录。空循环被优化为常量，不能用它估计
实际循环开销；这里报告调用组总时间，不作空循环扣除，也不将暖态开销直接扣减真实panel。
该结果不能覆盖记录buffer cache miss、首次调用、包装指令/cache扰动或worker相对到达变化，
因此不足以解释全部微秒级差异。下一改动优先隔离buffer记录与包装调度开销，不能直接
把换时钟视为解决方案。现阶段两个观测版本都不作为资源拟合数据来源。

本轮回顾：source唯一行为差分为相同指针避免重复release store，原子访问仍保持acquire/release，
原JIT getter每次照常调用；未改变CLI/日志schema/生产API/数值。原生烟测65调用与正式1066调用
全部数值/日志验证通过，clang-format、差分范围检查和文档diff检查通过。既有23个Python
观测/runner测试代码未变，沿用上一轮结果，不将其冒充本轮原子行为验证。尚未通过L3扰动门槛，
无采用/性能提升声明，未提交或推送。


## Worker分解及count-only消融（进行中）

对原版与缓存发布版各两场的既有worker trace做配对分解，逐call校验
`envelope=last_finisher_arrival_delay+last_finisher_interval`，只对均值使用可加分解。
缓存发布版24个M/stage/session组的到达偏移均值变化绝对值最大0.082us，多数阶段差异
为1–5us，不能将主要偏差归于team起点错位。`worker_perturbation_diagnostic.json`
同时保留逐worker平均区间变化，以及observe阶段区间扣除kernel区间和后的剩余量；
该剩余包含原wrapper/setup/barrier，不解释为纯观测开销。

新增仅Lab编译宏`JOINT_PANEL_COUNT_ONLY`，保留get_kernel/filter/sched_getcpu和每CPU计数，
关闭两次Now和64B事件payload写入。end_call仍验证正计数，但输出`joint_panel_count_only`
及`dispatch_count`，由独立`validate_counts.py`逐call/CPU核对预期拦截次数。原panel分析器
会因header不符拒绝这类记录，不能进入成本校准。默认未定义宏的完整记录逻辑保持不变。
新实验目录`tmp/joint_cost_model_20260911/panel_observer_count_only`；源、build、frontier、
protocol、run_smoke/run_paired、校验器及评分器均冻结。seeds611261/611262，其余8T真实
条带、NUMA、权重、5/31协议和1%门槛同前。E类Lab原生消融，不改变生产/Plan/API或模型系数。
当前构建/烟测/正式测量顺序执行中，只有前项成功才运行后项；不并行运行benchmark。


## Count-only消融完成：记录不是唯一剩余因素

原生O3/C++17/Wall/Wextra/Werror构建及65调用/1088拦截烟测通过；两场各533调用/10880拦截
通过数值、phase trace隔离和逐CPU计数校验。远端`run_paired.sh`返回COMPLETE，所有测量已结束。
本地`evaluate_paired.py`产出`panel_observer_count_only/perturbation.json`，两个session及compact
结果已取回，原始trace/计数日志保留远端同路径。现有23个观测/runner测试重新运行通过。

| M | W13 session1/% | W13 session2/% | W2 session1/% | W2 session2/% |
|---|---:|---:|---:|---:|
| 12 | +0.630 | +0.600 | +0.577 | +1.065 |
| 13 | +0.468 | +0.244 | +0.404 | +0.229 |
| 24 | +0.267 | +0.465 | +0.816 | +0.748 |
| 25 | +0.114 | +0.614 | +0.540 | +0.939 |
| 48 | +0.312 | +0.447 | +0.287 | +0.986 |
| 49 | +0.944 | +0.358 | +0.599 | +1.398 |

配对中位定义和1%门槛同前。22/24个GEMM组满足门槛；第二场W2/M12增加0.88us、
M49增加4.78us，仍未通过。gather配对变化-11.899%..+13.695%，整体target也有失败点；
不能将GEMM大多小于1%描述为整个观测方法通过。计数日志本身无时间戳，不可能作为panel成本输入。
三个库分别在自己的进程中与control配对，没有三库同进程随机干预，禁止将跨库差值当精确因果分摊。
结果说明仅去除时钟和event payload还不足以消除全部扰动；包装/CPU查询/计数及相对状态仍需区分。

保留count-only为有界诊断，不扩大硬件计数或更改已验证GEMM基线。下一候选减少同一次调用中
被观测的worker/panel数量，分别采样并保留完整未采样阶段对照；不得将非同时采样的worker区间
拼成一次真实team包络。1%门槛保持不变，真实共享需求/服务反馈/混合宽度/planner验收仍待推进。
静态复核确认新增行为只在Lab编译宏下启用，默认完整记录日志类型不变；无生产env/API/build改动，
未提交/推送。新增宏的原生集成已验证，完整记录分支复用既有原生数据并通过原解析器测试，
没有声称本轮重测了完整记录分支性能。


## 单worker轮换采样（进行中）

新增独立Lab编译宏`JOINT_PANEL_SPARSE_WORKER`，与count-only互斥。get_kernel解析原指针后，
仅CPU `312+call_id%8`进入观测包装，其他worker直接使用原指针。选择不读取M或权重地址，
但call顺序与pair/copy可能有关，须报告实际覆盖，不能默认统计独立。采样worker仍记录全部panel。
日志header为`joint_panel_sparse_worker`，分析器允许显式sampled_cpu但保留实际team宽度与N偏移，
不将其他7worker缺失作为完整记录，也不将这份数据拼为team包络。原完整记录和count-only语义保留。

源/编译/协议冻结于`panel_observer_sparse`，runner和新分析器单独冻结于`panel_observer_sparse_runner`。
seeds611281/611282，M12/13/24/25/48/49，NUMA3、实际8T窗口和5warmup/31pair同前；
`run_smoke.sh`通过才开始`run_paired.sh`。除原阶段1%门槛外，预先增加同pair同CPU的
被采样worker阶段区间1%门槛，防止观察开销被team其他worker最大值掩盖。`evaluate_sampled_worker.py`
报告每M/session的CPU及CPU/copy覆盖，并利用原全worker trace校验W13→W2阶段边界。
当前24个观测/runner测试通过，新增错误CPU、错误N偏移和稀疏数据不能伪装完整team的测试；
Ruff、clang-format通过。此为E类Lab采样诊断，不改生产/公共API/默认模型或拟合参数。


## 稀疏采样完成：team时间会掩盖被采样worker扰动

两场均COMPLETE，累计1066次native调用/2720条单worker panel记录通过数值、隔离、CPU/几何/
地址/块顺序校验；额外利用全worker native trace验证W13→W2阶段边界。24个定向测试、Ruff与
clang-format通过，新分析器也重新解析原版烟测65调用/1088panel，完整模式兼容。
`panel_observer_sparse/perturbation.json`与`sampled_worker_perturbation.json`保存两层门槛和覆盖。
每个M合并两场都覆盖CPU312–319，但32个CPU/copy组合仅覆盖25/30/29/27/29/27个，不声称全组合覆盖。

| M/stage | team session1/% | worker session1/% | team session2/% | worker session2/% |
|---|---:|---:|---:|---:|
| M12/W13 | +1.054 | +1.504 | +0.616 | +0.628 |
| M12/W2 | +0.702 | +0.337 | +1.220 | +2.615 |
| M13/W2 | +0.388 | +2.155 | +0.844 | +1.779 |
| M25/W2 | +0.589 | +0.951 | +0.303 | +1.278 |
| M48/W2 | +0.107 | +1.407 | +0.296 | +0.853 |
| M49/W2 | +0.447 | +0.909 | +0.737 | +1.307 |

24个GEMM组中team有22个满足1%，被采样worker只有15个满足。M13/W2 worker配对中位
绝对增加2.76/2.28us，第二场M12/W2增加2.08us；不得用team通过掩盖worker未通过。
gather/完整target仍有失败点，因此整个观测方法也不通过原门槛。三种既有观测对照及稀疏版
全部保留为有界诊断，不采用panel时间修正无竞争模型或拟合资源占比。

下一工作转向独立共享资源服务实验：固定已通过前瞻验证的whole-GEMM T0，先在相同实际
8T几何下做同LLC/跨LLC及背景强度对照，区分需求与响应，并独立验证资源容量。
这部分不依赖未通过的panel时间，不能将净历史曲线的a解释为计算下限，也不能据此宣称
panel成本/重叠比例已识别。真实panel观测缺口继续保留；未见动态、混合宽度和planner验收
仍属完整目标，未被这项诊断替代。生产kernel/JIT/模型默认值未变，无提交或推送。


## 真实路径locality/cohort对照（进行中）

`prepare_joint_locality.py`在原nested真实route上固定expert13/M13/8T，后台固定expert
37/233/151/157，声明M48/24/12/12并由native trace逐个校验。首批仅M13，不声称已覆盖
其他前台M。后台启用0/1/2/4个，local起始worker index40/48/56/64（CPU280–311），
cross起始index8/16/24/32（CPU248–279），前台index72/CPU312–319。sysfs本轮核实
CPU248 LLC域240–279、CPU312 LLC域280–319；均位于既定NUMA3。

所有条件保留同一五个head任务身份/宽度，后台不足4个时未激活head等待活跃cohort结束，
其他所有任务等待五个head完成，再按原任务依赖执行。原任务向量/allowed widths正确重排，
未更改生产PlanV2 schema或dispatch。local/cross的零背景各自配对，不能把核位置/初态差
统一吸收到竞争参数。后台运行真实gather/W13/W2，不是固定阶段或常驻stream；必须用trace
报告实际重叠与提前结束，背景数不是独立请求率，local/cross差也不是纯LLC/DRAM分离。

原native扩展、4份权重、216MiB scrub、固定workspace、H4096/F512、BF16/SVE256/Ntile16、
8T全条带(8,0,0,1,1)、W13/W2权重8/4MiB与owner1MiB/512KiB协议保持。无LD_PRELOAD或panel计时。
seeds611301/611302、两场5warmup/31pair。数据目录`tmp/joint_cost_model_20260911/locality`，
runner/分析器快照`locality_runner`，运行`smoke.sh`成功后才运行`run.sh`。实际输入为
`frontier_validated_rows.json`，早期`frontier.json`准备稿保留但未用于测量。

19个cohort/isolation定向测试通过：覆盖全部locality/count组合、拓扑顺序、静态核区间、
未激活任务与剩余任务依赖，以及非cohort重叠拒绝。Ruff通过。分析器复用原完整trace协议，
新增cohort隔离分支和真实背景M检查；没有关闭整体数值检查。当前原生烟测/两场采集顺序进行中，
尚无竞争响应结果或拟合系数变更。无竞争六类GEMM模型继续冻结。


## 首版cohort依赖错误与修复

首版烟测在local_n2/copy1失败，输出与workspace各393728个NaN，正式性能采集没有启动。
`locality/smoke.json`保存失败计划、runner/native/frontier身份；原错误计划和stderr保留。
静态复查确认提升后台到head时，仅保留了旧后继对head的依赖，却删除head的入边，
没有把原lane中的非head前驱连接到后继，导致剩余同核任务失去顺序。新资源校验器对旧
local_n2报`unordered overlapping worker intervals:56/68`；不能归因于生产kernel或噪声。

`cohort_bridge`现在递归穿过被提升head，把旧非head前驱依赖接回后继。额外构建DAG祖先集合，
验证所有共享worker区间的任务对都有先后关系，不仅检查拓扑序或head放置。
回归测试覆盖`remaining1→moved2→remaining3`的依赖保持，另加直接漏边拒绝测试。
20个cohort/isolation测试通过，新八个条件均通过全任务资源不重叠静态校验。

修复版独立目录`locality_spliced`和`locality_spliced_runner`，seeds611321/611322，其余条件
不变。新烟测成功后才执行新两场采集。`baseline_frozen.json`只记录已有六类模型SHA与M13
预测，没有读取本批测量拟合T0。`analyze_locality.py`报告每个locality自己的零背景偏差、
配对竞争增量、local-minus-cross增量差及实测阶段等效重叠；实测重叠只用于诊断，不能
冒充模型自主预测输入。当前修复版原生验证进行中，无有效性能结果声明。


修复版原生烟测通过，第一场36pair采集也已结束且数值检查通过。首次离线分析因
旧nested快照的`joint_worker_timing.py`缺少`initial_task_starts`导入失败；这不是测量失败。
原session1.json/trace保留，没有重测第一场。完整离线依赖另冻结到`locality_spliced_analysis`，
`resume.sh`先重放已有第一场trace，通过后仅运行session2。旧分析失败stderr保留。


## 真实M13 locality对照完成

修复版两场各369调用（45正确性前缀+324包含预热的采样调用），累计738调用通过数值、
真实foreground/background M、全call/phase完整性及非cohort排除校验，另有45调用独立烟测通过。
`resume.sh`最终COMPLETE。所有plan均关闭early merge，没有panel observer；背景实际N条带与
前台均为8T full stripes。原始trace留在Arm-codex-internal同目录，session/compact/analysis本地已取回。

冻结M13预测W13=267.5382us、W2=132.1967us。四个零背景条件会话实测中位W13=
264.83..265.56us，W2=131.23..132.06us；冻结预测偏差分别+0.745..+1.023%、+0.104..+0.737%。
本批基线没有显示足以解释下述约45us竞争增量的大误差；仍不重拟合T0。

下表为同session/pair相对各自locality零背景的配对增量中位数，单位us；不是两个中位数相减。

| 背景数 | local W13 s1/s2 | cross W13 s1/s2 | local W2 s1/s2 | cross W2 s1/s2 |
|---|---:|---:|---:|---:|
| 1 | 2.87 / 4.58 | 4.13 / 3.06 | 0.76 / -0.04 | -0.53 / 0.04 |
| 2 | 8.06 / 8.34 | 11.60 / 10.37 | 2.69 / 2.16 | 1.17 / 1.60 |
| 4 | 45.55 / 47.56 | 44.05 / 46.24 | 2.79 / 2.69 | 1.44 / 2.41 |

四后台W13绝对中位local311.87/313.38us、cross307.14/311.39us；W2为local133.85/134.17us、
cross133.84/133.73us。四后台W13配对相对增长约16.73..18.07%，W2约1.09..2.10%。
local-minus-cross竞争增量的逐pair差分中位W13四后台为+1.43/+2.29us，远小于两边共存的
约44..48us增量；两后台该差反为-3.73/-1.86us。不应将本批主要减速归为单纯同LLC额外惩罚，
但未计数实际DRAM/LLC服务，不能据此断言DRAM是唯一瓶颈或拟合出硬件容量。

实测动态重叠：四后台时，前台W13阶段平均等效后台约0.074..0.101个gather、
3.158..3.185个W13、0.555..0.567个W2；前台W2阶段则约1.347..1.417个W13和
0.578..0.646个W2（总数约2），两个M12后台大多已结束。配置N=4不能作为整段恒定压力。
W13/W2增量差同时包含背景阶段/活跃数变化和前台响应差异，本批不能单独识别各自贡献。

`analyze_locality.py`与`analysis.json`保留绝对时间、配对增量、差分和实测阶段重叠。
实测重叠为条件诊断，不是自主cost model的可用输入。下一步扩展前台M12/M48，同时补充
独立需求/服务证据，再比较实测时间线条件响应与自主动态推进；本批没有拟合系数、改变
planner基线或宣称完成联合泛化。20个定向测试、Ruff及静态diff检查通过；失败准备器与
离线导入错误均保留，未提交或推送。


## M12/M48真实cohort扩展与请求窗口候选

新增`--m12/13/48`（实际CLI为`--m 12`等，默认13），复用已修复的cohort DAG与全共享核
依赖检查；后台仍为expert37/233/151/157的M48/24/12/12，其他协议不变。M12 seeds611341/611342、
M48 seeds611361/611362，两个独立目录`locality_m12`/`locality_m48`及完整`locality_grid_runner`。
`run_locality_m_grid.sh`顺序执行两组烟测及四场采集，四场各369调用，合计1476次正式流程调用
通过数值、foreground/background M、资源cohort排除及完整trace校验，另90次烟测调用通过。
没有重复上一轮M13数据，没有panel observer或生产扩展重编译。

测量前，各目录`frozen_predictions.json`冻结M13 session1增量的两种静态迁移：绝对增量直接加到
目标T0，或把M13相对增量乘目标T0。不使用session2拟合，不截去训练中的微小负增量。
无竞争模型仍SHA50923778751e70fe9b630f24ac0f14fcac5239b53946cd335938a55e8e159e70，
M12 T0 W13/W2=161.18/82.485us，M48=624.855/320.12us。原native/source身份在各runtime记录。

同时实现`joint_burst_model.py`：每个stage用T0(min(M,12))作参考请求窗口，Q13/Q2为8/4MiB
packed-B代理需求，其余T0作为本资源之外的服务。请求窗口内按参考速率竞争共享容量，减速
延长窗口、改变后续重叠；W13结束后gap再进入W2，W2起点/终点不读取实测数据。仅各活跃team
首个W13起点是实测条件输入，当前不预测gather/初始到达，不能作为planner模型直接采用。
Q不是独立计数得到的DRAM流量，窗口不是已验证的逐panel物理分解；容量仅为有效模型参数。

纯容量候选仅用M13 session1拟合C=202.5（GB/s等效单位），训练16个条件阶段中位数MAE2.1843us。
M13训练中1/2后台已有低于饱和阈值的小减速，因此另拟合低压力响应版：
`u_i=1/(1+alpha*sum_peer(r)/C)`，再按总交付量施加容量约束；无背景u=1、T0不变。
得到C=200、alpha=0.08、gap=0.39us（gap来自零背景），训练MAE1.3914us。两个source快照和
profile分别冻结于`burst_candidate`/`burst_queue_candidate`，未修改基线或为新M增加系数。
9个事件/守恒/初态/零gap/参数域测试通过；低于单任务参考请求速率的容量会拒绝，防止破坏T0。

重要留出边界：静态迁移对照在新M采集前冻结；两个动态候选是在采集已经开始后、读取任何
M12/M48时间前冻结，之后才取回数据。这里称拟合留出检查，不将其冒充测前冻结的前瞻验证。
`fit_joint_burst.py`/`fit_joint_burst_queue.py`只读M13 session1，profile记录训练、baseline、model
source身份；`evaluate_joint_burst.py`从各冻结source快照运行，不使用被后续修改的活动代码。

## 条件动态模型的留出结果

以下只统计1/2/4后台的条件，排除零背景稀释；每个M为两session×两locality×三背景数。
双GEMM跨度从目标实测W13起点到预测/实测W2结束；cohort跨度从最早实测W13起点到所有
活跃expert W2结束，不包含预测gather/初始入口或剩余全计划任务。

| M/指标 | 纯容量 MAE/us | 低压力响应版 MAE/us | 后者MAPE/% | 后者P90相对绝对误差/% |
|---|---:|---:|---:|---:|
| M12/W13 | 3.325 | 2.626 | 1.442 | 2.739 |
| M12/W2 | 1.080 | 0.422 | 0.493 | 1.168 |
| M12/目标双GEMM跨度 | 4.125 | 2.482 | 0.925 | 1.996 |
| M12/cohort跨度 | 3.919 | 3.344 | 0.337 | 0.707 |
| M48/W13 | 6.360 | 5.444 | 0.838 | 1.927 |
| M48/W2 | 0.683 | 1.745 | 0.546 | 0.874 |
| M48/目标双GEMM跨度 | 6.466 | 6.193 | 0.639 | 1.288 |
| M48/cohort跨度 | 7.116 | 5.653 | 0.570 | 1.424 |

低压力版本没有每项都改善：M48/W2绝对误差比纯容量版本更大。M12/W13最大单组误差仍为
3.513%，不能说所有阶段组均低于3%。M13 session2重复检查中，低压力版W13/W2 MAPE为
0.661%/0.609%，目标/cohort跨度为0.568%/0.360%；这不是未见形状结果。

测前静态迁移对照在有竞争条件的W13 MAE：M12绝对/相对迁移3.970/4.736us，M48为
6.329/31.959us；W2分别1.358/1.313us、1.970/4.200us。整段相对slowdown在M48上迁移较差。
动态候选额外使用实测初始W13起点，所以这不是输入信息相等的完整planner对照，不能据此
宣称全模型优于静态版本。`locality_transfer_evaluation.json`保存静态评估，两动态目录的
`evaluation.json`保存全部分组/有符号残差，负例不删除。

本轮保留低压力请求窗口版作为下一步候选，纯容量及静态两版保留对照，默认planner不变。
下一关键验证是接入gather/worker-ready预测，移除实测W13起点，再用新鲜条件检查全时间线；
同时补独立需求/服务证据，不能把C=200解释为已测硬件峰值。其他M/阶段历史、混合宽度、
真实全trace、固定候选选择与相同预算planner搜索仍未完成。无提交/推送。


## 移除实测W13起点：入口/gather组合

新增`joint_ready_burst.py`，只接收初始无依赖的核不重叠8T jobs，以及已校准的入口/gather/
GEMM/resource数据，不接收被预测运行的时刻。每worker的gather服务由既有四特征系数算出，
team gather结束=max(A_i+G_i)，再加M13 session1零背景测得的gather→W13间隙2.28us。
后续请求窗口模拟的C=200、alpha=0.08、W13→W2 gap=0.39us保持冻结。
原始入口样本来自`initial_ready/context_session1.json`的31个完整80CPU anchor向量；
旧per-CPU median是对照，主候选保留每次80CPU相关性，模拟31个场景后取输出中位数。

当前无gather竞争反馈，不支持任意DAG/混合宽度/其他NUMA。非8T、共享核cohort或非空deps
会被拒绝，避免把此候选静默当完整planner；依赖任务的后续ready模型仍需单独接入。
14个ready/burst/旧worker-ready测试通过，包括联合向量保留、先gather后GEMM、资源域拒绝。
Ruff通过。M13第一场仅用于原resource及独立阶段gap，未用M12/M48重新拟合参数。

回顾性结果（M13只列session2，M12/M48两场；包括零背景）：

| M/指标 | per-CPU median MAE/us | correlated MAE/us | correlated MAPE/% |
|---|---:|---:|---:|
| M13/目标完成 | 8.413 | 3.967 | 0.705 |
| M13/cohort完成 | 12.012 | 7.409 | 0.789 |
| M12/目标完成 | 6.363 | 7.331 | 1.805 |
| M12/cohort完成 | 10.878 | 6.670 | 0.818 |
| M48/目标完成 | 30.961 | 24.204 | 2.085 |
| M48/cohort完成 | 33.098 | 25.640 | 2.203 |

不是每项都改善，M12目标完成MAE略回退。M48入口偏差较大：两场合并全部条件的中位最晚
worker到达141.43/184.52us，max worker gather区间25.31/24.05us。用实际到达+预测gather+固定
setup作条件诊断，中位W13起点误差-1.70/+0.24us，说明此次共同入口延后主导第二场起点偏差，
不能直接归于gather斜率或拟合一个M48专属偏移。完整分组见`ready_burst_candidate/retrospective.json`。

## M17/M35前瞻验证启动

`ready_m17`/`ready_m35`各八个cohort条件+anchor，后台/摆放/8T条带/NUMA3/四副本/
216MiB scrub/5warmup31pair/early_merge off均沿用。准备器允许M17/M35，使用同expert13的
真实route前缀，并再次静态检查全部共享核任务对的DAG先后。完整预测先写入各目录
`frozen_predictions.json`，包括主版本correlated及per-CPU median对照；其后才运行
`run_ready_validation.sh`。seeds611381/611382、611401/611402，预测源/输入身份均保存。

验收从scheduled_compute起点到前台/cohort W2完成，含预测入口和gather；目标/cohort
MAPE<=3%、P90绝对相对误差<=5%，另报告阶段起点/时长失败及有符号偏差。不得在看到新数据
后改选主版本或用实测起点重算预测。场景P10/P90仅是旧入口向量分布，不当作已校准预测区间；
另报告覆盖率。当前M17第一场已验证，剩余会话顺序运行，尚不能宣称前瞻通过。


## M17/M35新鲜无实测起点验证完成

四场各369调用，累计1476调用及另90次独立烟测通过数值、真实M、完整trace和非cohort
排除校验；`run_ready_validation.sh`最终COMPLETE。原始trace位于Arm-codex-internal同目录，
紧凑数据与运行身份本地已取回。`evaluate_ready_validation.py`只读取各目录测前冻结的数值
预测进行评分，没有按验证实测重算起点、挑选场景或重拟合参数。主版本保持correlated。

以下每M为两场×两locality×四背景数共16条件会话，包括零背景；完成时间从scheduled_compute
起点计，已包含预测的worker入口、gather、阶段衔接和GEMM竞争。

| M/指标 | MAE/us | MAPE/% | P90相对绝对误差/% | 最大相对绝对误差/% |
|---|---:|---:|---:|---:|
| M17/前台完成 | 8.665 | 1.485 | 3.159 | 3.961 |
| M17/cohort完成 | 10.489 | 1.134 | 2.130 | 2.400 |
| M35/前台完成 | 9.667 | 1.073 | 1.923 | 2.287 |
| M35/cohort完成 | 9.908 | 0.935 | 1.640 | 1.725 |

四个完成时间分组均通过预设MAPE<=3%、P90<=5%的本轮门槛，没有将标准改成单次误差门槛。
GEMM阶段MAPE：M17 W13/W2为0.685%/0.374%，M35为0.921%/0.288%。W13起点MAE仍为
8.570/6.542us，方向分别偏晚+8.479us、偏早-5.032us；不把完成误差较小解释为入口已经准确。

per-CPU median对照同样通过完成时间门槛：M17前台/cohort MAPE1.572%/1.155%，M35为
2.232%/1.746%。预先指定的correlated版本在本轮总体更稳，但M17起点MAE反而较大，
不能声称每个指标都优于对照。所有分组和方向保留于`ready_burst_candidate/prospective.json`。

旧入口场景P10/P90带对前台样本覆盖率：M17为69.2%、M35为78.2%；cohort为58.5%/71.2%。
这些场景没有包含全部kernel服务噪声和会话漂移，阶段时长覆盖率更低，因此不作为已校准
完整预测区间或剪枝依据。该限制不通过事后扩大范围或删除异常样本来掩盖。

本轮接受“初始无依赖8T cohort、声明M范围内”的无实测起点候选作为后续实验基础；默认
planner仍不切换。下一步接依赖驱动的后继任务及gather衔接，验证完整队列顺序；之后扩展
大M和其他宽度、独立需求/服务证据与真实全计划/相同预算搜索。初始cohort验证没有替代
完整目标，未提交或推送。14个组合模型定向测试与Ruff通过，数学模型/manifest已同步范围和限制。


## 接入依赖后继与五条队列（进行中）

`joint_burst_model.simulate`新增可选prepare回调与按expert引用的拓扑deps，父任务W2结束后
才释放子任务；`joint_dag_burst.py`先验证所有共享CPU任务对存在祖先顺序，再用
`max(initial_entry, parent_ready)+gather_service`准备下一GEMM。不重复收取初始入口，
多父节点等最晚父节点。原无deps接口保留，无依赖cohort与旧组合预测逐字段一致。
不重拟合C=200/alpha=0.08/gap=0.39、gather/setup=2.28或入口场景，M域仍<=60、宽度8T。

17个DAG/burst/cohort/queue测试通过：root等价、同核后继、跨lane join、非法依赖/共享核、
以及提升pilot任务后保留剩余lane链；Ruff通过。准备器`prepare_joint_queues.py`使用真实
原route（expert13仍M60），五lane CPU280–319，每条三expert，固定分配如下：
`[13,26,12]`、`[37,29,36]`、`[19,136,81]`、`[211,245,70]`、`[156,233,24]`；
相应M为`[60,12,8]`、`[48,24,8]`、`[35,35,12]`、`[36,32,12]`、`[40,24,16]`。
forward/reverse/rotate/staggered只改变各lane顺序；joins和joins_reverse另加每层全局汇合，
serial将15个pilot任务全串行，均为明确对照。剩余原expert等pilot完成后再按原lane依赖执行。

实验目录`tmp/joint_cost_model_20260911/queues`、runner `queues_runner`；seeds611421/611422，
8计划含anchor，每场5warmup/31pair，其他NUMA3/4权重副本/216MiB scrub/8T full stripes/
固定workspace/early_merge off协议同前，无panel observer。所有计划资源对均静态验证。
`frozen_predictions.json`测前冻结所有完成与expert端点预测，预计四个队列顺序相差约37us，
需要配对顺序差验证，不能只看约2ms总时间误差。当前两场已结束并通过基础trace/数值校验，
正在取回数据做冻结评分与逐依赖边复核。

`queue_phase_diagnostics.py`在测量开始后仅用同一冻结输入推导每场景的阶段时长中位数，并
精确复现所有测前完成预测；未拟合或读取新实测时刻。其独立`phase_predictions.json`避免
用两个端点中位数之差冒充阶段时长中位数，完成时间验收仍只用测前文件。


## 稠密队列失败与运行时发布成本

首批两场各328调用（40正确性前缀+288含预热采样），共656调用通过数值、实际M及基本trace。
`evaluate_queues.py`进一步检查每个pilot依赖边、gather→W13→W2顺序，均通过。四个无汇合队列
完成MAE89.958us、MAPE4.273%、P905.196%，未通过；joins为MAPE1.902%，serial为0.947%。
全部预测及有符号误差保留在`queues/evaluation.json`，没有改门槛或加统一补偿。

关键lane与预测一致（forward末expert24，其他三个顺序末expert156，两场每轮均为该lane）。
主要遗漏发生在后继gather之前：稠密chains同lane后继通常等待约40–50us，gather到达跨度
只有约1us；每lane两次交接即可形成约90us低估。joins只在上层最晚完成者自己的lane出现长等待，
其他空闲lane约2–5us；serial轮换lane通常1–3us。这不是所有后继重复初始到达，也不能只按任务数加罚。

只读原native源码确认：`run_async_task`在W2结束后barrier，leader先store task state2，
依次对所有successor执行`deps_remaining.fetch_sub`，之后更新expert/completed状态再做team barrier。
本实验为隔离剩余工作，给每个pilot都连接约209个非pilot，早期pilot子任务先被通知，但原team
须等待全部通知完成才可复用；其他lane可能早已空闲。因此必须区分data/dependency-ready与
worker-reusable。源码依据`csrc/moe/arm/common/fused_moe_bf16_tiled.cpp`约8868–8890行。
原anchor实际224任务/215边，此大fanout主要由实验隔离图引入，不据此给所有8T加固定40us惩罚。

## 保持逻辑顺序不变的传递约简配对

新增Lab `joint_dag_reduction.py`，计算每个任务完整祖先集合，删除已被其他父节点覆盖的边，
逐节点检查约简前后祖先集合完全相等；不改task顺序、核、宽度、窗口、输入或kernel。
原图保留，未全局修改生产planner。独立8个图/队列/DAG定向测试通过，Ruff通过。

`queue_reduction`含forward/reverse/staggered/joins/joins_reverse/serial各dense/reduced一对，
加anchor共13计划，同进程、同pair/copy对照。seeds611441/611442、5warmup/31pair，
两场1066调用及另65调用烟测均通过数值、实际M、完整trace/隔离校验。边数：普通队列3345→255，
joins3385→295，serial3349→223。Pilot内部deps完全相同，复用原测前prefix数值预测；没有参数拟合。

同lane且自己的前任务为当前关键依赖时，gather前间隙的条件中位数汇总从43.47us降到3.01us
（dense范围38.59..54.60us，reduced2.31..7.80us）。直接配对完成时间收益如下，正值表示约简更快：

| 计划 | session1 dense-minus-reduced/us | session2/us |
|---|---:|---:|
| forward | 78.26 | 92.77 |
| reverse | 67.42 | 68.49 |
| staggered | 90.02 | 86.80 |
| joins | -4.22 | -2.26 |
| joins_reverse | -18.85 | -3.02 |
| serial | 1.97 | 5.46 |

三队列差的pair MAD约21..26us，收益在两场方向一致；joins/serial差相对各自MAD较小，
不宣称所有约简计划都有稳定加速。与未改变的模型预测比较：

| 分组 | dense MAPE/% | reduced MAPE/% | reduced P90误差/% |
|---|---:|---:|---:|
| 三条队列顺序 | 4.413 | 0.787 | 1.537 |
| 两种joins | 1.240 | 1.242 | 1.725 |
| serial | 1.234 | 1.188 | 1.203 |

例如reduced forward实测2025.52/2049.24us、预测2015.797us；reverse2032.45/2054.10us、
预测2033.106us；staggered1993.52/2024.97us、预测1995.785us。这证明冗余通知解释了主要
队列差距，不等于原模型已能预测任意稠密表示；实际改变的是实验图表示，不是cost系数。
`queue_reduction/evaluation.json`保存每plan绝对值、配对差、MAD及交接分类。

## 仍需独立处理的问题

混合首波M8/W13在原队列中约低估10–15us。约简后expert12/36的reverse阶段仍约192.68..196.15us，
同进程serial分别约117.39..121.61us，而基线M8为125.44us；因此不能用约简或调高基线解释
竞争下的全部偏差。共享窗口的带宽分配/小M响应仍需针对性验证，不加入M8常量残差。

原四队列顺序差的误差约0.7..22.1us，pair MAD约13..33us；个别对照跨场变号，不把噪声内
顺序差当作可靠排序。此15-expert前缀没有覆盖完整原计划、M>60或混合宽度。
后续明确分开依赖通知时刻与worker可复用时刻，使用后继通知工作量刻画发布成本，并保持
稠密/约简反例与M8竞争问题分开验证。默认planner、native/kernel和模型参数未改变；未提交/推送。


## 发布成本接入模型与中间fanout验证

`joint_burst_model`新增可选completion回调；`joint_dag_burst`在启用publication时要求每个job
携带完整native successor顺序，包括不在15-expert前缀里的任务。父任务W2结束后，依次计算
每条边通知时刻E+b+nu*rank，原team可复用时刻E+b+nu*successor_count。子任务逐worker同时
等待逻辑通知和该worker的可复用时间，初始entry仍只计一次。默认publication=None保留旧结果。
不把发布期间的原子访存伪装成已建模的DRAM流量；本轮只显式处理时序/占用，资源参数保持冻结。

`fit_publication.py`仅读取queue_reduction/session1的forward/reverse/staggered同lane交接，
各图各10个后继，按条件中位统计。fanout1/210的中位交接为2.955/44.115us，拟合b=2.7580622us、
nu=0.1969378us/通知。该线性形式尚需中间点验证，不解释为每条atomic的独立硬件延迟。
profile与全部训练点保存在`publication_candidate`，相关模型/依赖源文件已另存`model/`快照。

新增`expand_pilot_fanout`仅添加已经由祖先关系隐含的边，逐次约简检查可达关系完全一致。
从约简图构造每个pilot恰好9/33/65个successor，分别用于forward/reverse/joins/serial；共12条件
加anchor，保持pilot依赖/核/输入/工作量不变。完整图边数为335/695/1175。18个图、DAG、burst、
cohort测试通过，覆盖早通知与晚复用分离、未建模successor仍占用发布时间、元数据缺失拒绝
和扩展边不改变可达性。Ruff通过，无native代码或生产配置改动。

`publication_validation/frozen_predictions.json`在采集前冻结新模型与旧模型数值；seeds611461/
611462，13计划、每场5warmup31pair，源/协议身份保存。预计forward完成随fanout9→65由
2024.52→2046.14us，reverse2040.96→2060.11us；serial各点8605.05us，不将fanout退休成本
错误累加到闲置足够久的下一lane。joins预计2660.90/2660.60/2658.68us，保留潜在竞争错峰效应。
新图数值烟测通过后顺序两场采集，目前进行中；不会用实测时刻重算预测或重新拟合参数。


## 中间fanout前瞻结果：完成时间通过，线性交接成本仍偏低

两场各533调用、合计1066调用及另65次烟测通过数值、实际M、完整trace/隔离校验，
`publication_validation/run.sh`最终COMPLETE。所有评分只使用测前`frozen_predictions.json`。
没有读取新时刻重新启动模型，没有用验证结果重新拟合。19个发布/DAG/burst/图测试通过，
新增晚序号通知延后另一team、未建模successor影响占用、零发布成本退化为旧模型的验证。

每组为fanout9/33/65×两场，共6条件；completion从scheduled_compute计到15个pilot W2结束：

| 分组 | 旧模型MAE/us | 发布模型MAE/us | 旧MAPE/% | 发布MAPE/% | 发布P90误差/% |
|---|---:|---:|---:|---:|---:|
| forward | 65.051 | 46.044 | 3.119 | 2.209 | 2.684 |
| reverse | 51.472 | 34.516 | 2.463 | 1.653 | 2.195 |
| joins | 13.081 | 18.894 | 0.496 | 0.717 | 1.218 |
| serial | 118.522 | 79.909 | 1.365 | 0.920 | 0.971 |

发布版四组均通过本轮MAPE<=3%、P90<=5%的完成门槛，forward由失败转通过；joins略回退，
不声称每项改善。示例：forward_n65预测2046.14us，实测2097.21/2107.12us，仍有51.07/60.98us
低估；serial三fanout均预测8605.05us，实测8680.43..8692.48us，没有随fanout大幅增长。
保留这一行为区别，不能换成每任务统一退休时间相加。

单项交接诊断仍揭示线性插值不足。对forward/reverse同lane后继，预测gap就是b+nu*N，
其模型端点差在此为常量；实际按每个条件/后继取31轮中位，再汇总：

| fanout | 预测gather前gap/us | 实际gap中位/us | 逐条件gap MAE/us |
|---|---:|---:|---:|
| 9 | 4.531 | 4.955 | 0.614 |
| 33 | 9.257 | 12.130 | 3.146 |
| 65 | 15.559 | 23.460 | 8.080 |

不能用较长完成时间的门槛通过掩盖该局部低估。当前只支持通知/worker复用分离的结构和
此次完成时间结果；线性按条数收费仍是待改进代理，通知对象布局/并发状态尚未独立辨识。
没有事后将9/33/65并入拟合，也没有加M或计划残差。M8/W13竞争响应低估仍单独保留。

产物`publication_validation/evaluation.json`保存所有计划绝对值、分组误差和交接诊断；
原始trace留在Arm-codex-internal同目录，compact/身份已本地保存。runtime代码未修改，
新增行为仅在Lab模型的显式publication参数下启用，旧模型对照保留。下一步继续处理通知成本
特征与M8资源分配假设，并扩展大M/宽度/完整计划；默认planner未切换，无提交/推送。

## 小 M 共享分配规则的条件诊断

离线命令：`.venv/bin/python tmp/joint_cost_model_20260911/allocation_diagnostic/analyze.py`。
使用queue_reduction两会话dense/reduced reverse及joins_reverse，首波五个8T expert的
M为8/8/12/12/16。固定原T0、C=200、queue_scale=0.08、gap=0.39us；只切换
proportional与max_min（等宽team公平共享，低需求先满足）。没有重新拟合参数。
输入使用当前样本的实际W13入口，其后阶段自主推进，属于回顾性条件诊断。

首次检查发现不能把全部首波W13都当成无后继干扰：M16较长，其他lane可能已启动下一任务。
因此逐目标仅保留实际及两种预测W13结束都早于首个非首波gather的共同样本。
M8/M12/M16分别保留493/495/129个expert-call；M16存在明显筛选，不能代表全部M16。
诊断脚本和results.json保留逐样本数据、每条件样本数、源文件SHA及范围说明。

| M | proportional有符号误差/us | max_min有符号误差/us | proportional MAE/us | max_min MAE/us |
|---|---:|---:|---:|---:|
| 8 | -15.387 | +11.463 | 15.387 | 11.463 |
| 12 | +33.446 | +8.364 | 33.446 | 8.364 |
| 16（筛选子集） | +11.433 | -13.226 | 11.433 | 13.226 |

统计先取每个session/plan/expert的逐样本预测减实测中位数，再等权汇总条件；
MAE是这些条件误差绝对值的均值，不是两个独立端点中位数之差。
比例分配同时低估M8、高估同波M12，说明仅提高M8无竞争成本不能解决该混合波次。
max_min明显缩小M12偏差，但M8反向高估，M16筛选子集略变差，不能直接采用或声称
证明硬件公平共享。请求窗口/计算访存重叠仍可能与分配规则共同贡献误差。
下一步需冻结候选，在新小M、同类及异类背景、启动偏移上辨识这些假设，不能按本表
调插值系数后仍称独立验证。默认simulate仍为proportional，planner未切换。

定向验证：`.venv/bin/pytest -q tests/test_moe_joint_burst_model.py tests/test_moe_joint_dag_burst.py`
为15 passed；本轮无新增远端采集，无native或production改动。

## M4/M8新鲜竞争对照已冻结并启动

继续M/E类Lab验证，保持production/default不变。准备命令：
`.venv/bin/python tmp/joint_cost_model_20260911/prepare_allocation_small.py`。
复用已验证ready_m17真实cohort DAG，仅将同一个expert13的嵌套route前缀改为M4/M8，
各自8条件+anchor：背景M48/M24/M12/M12，0/1/2/4背景，同LLC与跨LLC。
所有bridge再次通过共享worker祖先关系校验，实际M由原runner的nested-route与trace校验确认。

每个条件对proportional/max_min两种规则分别冻结旧31个相关入口场景的自主预测，
无竞争成本、C/alpha/gap、gather系数均不改。比例版逐场景结果断言与旧cohort接口完全一致。
冻结包含逐expert双GEMM时长、完成端点、cohort端点及全部场景，避免两个端点中位数相减。
模型源码与profile/基线另存`allocation_small_m/model`、`profile.json`、`baseline.json`。

远端Arm-codex-internal `/home/zhangxu/codex/fused_cpp`，沿用NUMA3/CPU240–319、8T full
stripes `(8,0,0,1,1)`、Ntile16、W13/W2 8/4MiB、owner 1MiB/512KiB、四权重副本、
216MiB scrub、固定workspace、early merge off、5warmup+31pair。扩展SHA与既有版本一致。
seeds为611481/611482与611501/611502；调用
`bash tmp/joint_cost_model_20260911/allocation_small_m/run.sh`顺序执行四会话。
启动前确认没有其他同类benchmark进程。当前采集进行中，不作通过或改善声明。

进度：M4两会话已由原runner完成数值/实际M/trace校验，M8随后串行采集。
离线追加M1/4/8/12/13/24/48/60各1/2/5个同步同形任务，共24个cohort，
两种分配所有事件端点在1e-8us内一致；没有修改模型文件。评分器
`allocation_small_m/evaluate.py`仅读测前预测，分零背景/有背景报告前台W13/W2、
前台完成和cohort完成，并保留每条件MAD、背景及locality，防止长后台隐藏小M局部误差。

## M4/M8前瞻结果：低压力响应也不足，不能只替换饱和分配

上述四会话均终止成功，各369调用、合计1476调用通过数值/实际M/trace校验。
原始trace留在远端同目录，session/compact已本地取回。评分命令：
`.venv/bin/python tmp/joint_cost_model_20260911/allocation_small_m/evaluate.py`，
结果`evaluation.json`检查frontier、seed、模型快照身份、31个非warmup pair及数值/isolation状态。
全部使用测前预测，未重新拟合。

零背景M4/M8的W13 MAPE为0.465%/0.509%，W2为1.209%/0.771%，基线没有同量级漂移。
M8零背景完成却低估19.683us/MAPE5.499%，说明入口/gather/间隙仍有独立误差，不应吸收入GEMM。
有背景12个条件会话（local/cross×1/2/4背景×两场）结果：

| 前台 | 指标 | proportional | max_min |
|---|---|---:|---:|
| M4 | W13 MAE/us | 14.295 | 10.917 |
| M4 | W2 MAE/us | 4.093 | 4.653 |
| M4 | 前台完成MAPE/% | 5.108 | 3.670 |
| M4 | cohort完成MAPE/% | 1.140 | 1.425 |
| M8 | W13 MAE/us | 11.218 | 8.952 |
| M8 | W2 MAE/us | 2.365 | 4.751 |
| M8 | 前台完成MAPE/% | 8.469 | 6.515 |
| M8 | cohort完成MAPE/% | 2.100 | 2.460 |

两种规则在有竞争前台完成均未通过MAPE<=3%、P90<=5%的门槛；cohort完成全部通过，
但最长后台会隐藏小M局部失败，不能据此采用。公平分配的W13平均偏差接近零只是正负抵消：

| M | 背景数 | 实测W13/us | proportional/us | max_min/us |
|---|---:|---:|---:|---:|
| 4 | 1 | 124.36 | 120.00 | 120.00 |
| 4 | 2 | 135.02 | 122.31 | 122.31 |
| 4 | 4 | 184.38 | 158.56 | 200.06 |
| 8 | 1 | 130.89 | 127.71 | 127.71 |
| 8 | 2 | 139.90 | 130.20 | 130.20 |
| 8 | 4 | 187.92 | 167.15 | 201.90 |

上表先取每条件31轮中位，再平均local/cross与两场，不能解释为硬件带宽测量。
1/2背景在当前模型内未触发容量分配差异，两种规则却均低估，故仅修饱和时公平性不足。
另执行`allocation_small_m/conditional.py`，读取实测初始W13入口做独立条件诊断，不改变冻结评分。
1/2背景W13条件偏差M4仍为-4.497/-12.938us，M8为-3.132/-9.860us；
4背景比例/公平偏差M4为-28.779/+10.875us，M8为-22.384/+11.205us。
入口误差不能解释低压力W13响应不足，未来应固定T0，用这些开发数据辨识前台敏感度与
请求窗口/共享服务，参数冻结后另采新M/混合比例验证；本批不再充当新拟合的独立验证。
保留两候选，默认不切换。完整宽度、大M、实际全计划与planner regret仍未完成。

## 前台敏感度可辨识性与入口校准诊断

离线脚本在`sensitivity_diagnostic`，使用已冻结allocation_small_m模型快照；没有改默认接口。
`run.py`仅允许目标W13的alpha变化，后台、W2、T0、C、gap和比例/公平规则均保持；
分别用M4/M8 session1的n1或n2条件拟合，给其他背景数与session2评分。全部初始W13入口来自
实测，因此是开发诊断，不是自主或新鲜验收。0..2网格粗步0.02、局部细步0.001，无新增依赖。
原alpha=0.08的候选在全部观测样本上逐字段复现旧模型。

| M | 训练背景数 | alpha | session2 n1 W13偏差/us | n2/us | n4/us |
|---|---:|---:|---:|---:|---:|
| 4 | 1 | 0.261 | -0.014 | -3.640 | -8.322 |
| 4 | 2 | 0.324 | +1.573 | -0.288 | -1.656 |
| 8 | 1 | 0.218 | +1.480 | -1.075 | -5.669 |
| 8 | 2 | 0.261 | +2.682 | +1.407 | -0.628 |

表用原比例分配；提高前台敏感度已明显改善n4，无需立即改公平分配。
但n1/n2识别的系数不同，单线性响应仍可能遗漏曲率，不能把每个系数当硬件常数。
`run_extended.py`对旧M12/M13相同对照的n1/n2识别alpha分别为0.081/0.115、0.028/0.102，
提示不能给所有M统一加相同敏感度。历史session2仍有波动，完整残差保存在extended_results.json。

`entry.py`只看M4/M8零背景，以逐worker实际gather起点代入冻结gather服务，精确分解W13入口
延后=到达项+gather服务残差+setup残差，使用均值保证可加（不同于冻结中位数评分）。
M8两场平均入口延后29.073/29.680us，其中到达项29.051/29.705us，gather残差
-0.203/-0.231us，setup+0.224/+0.206us。到达项含dispatch与worker readiness，非纯唤醒因果测量。
M4入口延后-0.940/+14.004us，也主要来自到达项。

`anchor_entry.py`使用同进程pair5..14的10次独立anchor完整80CPU入口向量，预测pair15..35
的零背景前台；未使用目标时刻校准。M8这21轮子集的完成MAE由旧19.166us降至1.964us，
M4为7.748→6.990us。此为回顾性校准策略诊断，不替换原前瞻分数；策略还需新会话验证。
不能给M8直接加固定20us，也不能把同进程校准收益解释为跨会话到达模型已泛化。

## 单参数请求密度敏感度候选与M6/M10留出

保留原比例分配和全套基线，只令W13请求窗口的
`alpha_i = 0.08 + beta * max(0, T0_W13(12)/T0_W13(min(M_i,12)) - 1)`。
比值等于既有packed-B首窗口参考请求率相对M12的比例，是需求代理，不是独立DRAM实测。
仅用M4 session1、n1/n2、local/cross四条件的条件W13残差拟合一个beta，得到0.632，
训练RMSE1.207us。隐含M4/M6/M8/M10/M>=12的alpha为0.31185/0.29780/0.26007/0.14907/0.08。
无M逐点新参数，不改变W2响应、后台M>=12、容量或请求量。beta=0恢复原模型，
M1/4/6/8/10/12/13/48/60九个隔离形状全部逐字段保持T0。
这一密度与敏感度关系仍是假设，不能凭两个小M解释成功认定物理机理成立。

准备命令`.venv/bin/python tmp/joint_cost_model_20260911/prepare_density_small.py`。
`density_small_m`保存完整源码、profile、beta训练记录、基线和测前场景预测；新前台M6/M10
各8条件+anchor，背景和local/cross协议同allocation_small_m。seeds611521/611522、611541/611542。
默认预测仍使用旧独立入口场景；入口校准策略与GEMM候选分开，未事后加入冻结预测。
远端执行`bash tmp/joint_cost_model_20260911/density_small_m/run.sh`，启动前核实无重叠benchmark、
扩展身份与原协议相同，当前顺序采集中。原始trace留远端，评分器evaluate.py仅读冻结预测。
本轮是M/E Lab候选，生产、默认planner、kernel未变；未提交/推送。

## M6/M10冻结验证：密度修正迁移到M6，但M10回退

`density_small_m/run.sh`四会话全部终止成功，各369调用、共1476调用通过原runner数值、
实际M、trace/isolation校验。session/compact及runtime身份已本地取回，原始trace保留于
Arm-codex-internal同路径。参数beta=0.632未改，所有评分仅读取测前frozen_predictions.json。
命令：`.venv/bin/python tmp/joint_cost_model_20260911/density_small_m/evaluate.py`。
零背景M6/M10的W13 MAPE1.079%/0.794%，W2 0.447%/0.684%，没有竞争误差同量级基线漂移。

有竞争为每M的12条件会话（local/cross×n1/n2/n4×两场）：

| M | 指标 | 原比例模型 | 请求密度敏感度候选 |
|---|---|---:|---:|
| 6 | W13 MAE/us | 14.295 | 1.135 |
| 6 | W13 MAPE/% | 8.854 | 0.734 |
| 6 | W2 MAE/us | 2.951 | 4.250 |
| 6 | 前台完成MAPE/% | 11.729 | 8.942 |
| 6 | cohort完成MAPE/% | 3.113 | 3.335 |
| 10 | W13 MAE/us | 1.876 | 4.431 |
| 10 | W13 MAPE/% | 1.076 | 2.664 |
| 10 | W2 MAE/us | 1.029 | 0.672 |
| 10 | 前台完成MAPE/% | 5.044 | 3.923 |
| 10 | cohort完成MAPE/% | 2.191 | 2.263 |

候选有竞争前台完成两组均未通过MAPE<=3%、P90<=5%门槛；M6 cohort也未通过。
M6 W13明显改善但W2回退，M10 W13反向高估4.431us，不采用通用修正，也不事后调beta。
候选减少平均前台完成低估并不能覆盖上述阶段回退。

`conditional.py`保持beta固定，只代入实测初始W13时刻作诊断：M6 n1/n2/n4的W13
偏差原为-4.539/-12.718/-28.876us，新为+0.976/-1.035/-5.828us；M10原为
+0.610/-1.029/-6.223us，新为+2.811/+3.649/+3.392us。
因此M6自主W13的1.135us MAE不能全部归功于竞争模型准确：n4条件诊断仍有约5.8us低估，
入口/重叠误差可能部分抵消。M10多压力系统高估也不是单纯入口偏移。

`increments.py`逐pair比较同会话同locality的n1/n2/n4与n0阶段时长，避免两个不配对中位数相减。
M6 W13增量为5.600/16.183/65.298us，M10为2.050/6.433/46.968us；相应pair MAD
M6为1.290/1.470/3.230us，M10为1.305/1.560/2.800us（均为四条件统计的均值）。
需求率线性映射不能完整表达这两个M的响应差异。计算/访存可隐藏部分是否造成响应转折是
下一候选假设，尚非硬件因果结论；不同kernel行类别、W13/W2敏感度应分开检验，不能据此硬编码M阈值。

入口策略迁移诊断`../sensitivity_diagnostic/anchor_entry_transfer.py`沿用既定pair5..14
独立anchor预测pair15..35零背景规则，M6 MAE27.785→10.323us，M10为18.575→8.471us。
有改善但未达到M8旧诊断的1.964us，不能宣称入口问题已解决，也不改写冻结评分。
此策略脚本在本批时间数据取回前准备，但非测前冻结数值预测，保留校准策略诊断口径。

旧混合队列回放`queue_diagnostic.py`也保持beta固定。共同有效样本上M8 W13 MAE
15.387→2.709us、M12 33.446→17.643us、M16筛选子集11.433→4.997us；
M12仍显著高估，未闭合全部共享服务。它是已观察数据的条件诊断，不是新鲜验收或全计划结果。
完整逐条件、样本噪声、冻结预测和诊断分别保存在evaluation/conditional/increments/queue_diagnostic.json。
当前无远端采集运行，默认planner未切换；下一步需验证重叠/响应转折，并保留本批反例。

## 独立计算/请求时钟的重叠候选：离线辨识未支持采用

本轮无新远端采集。`overlap_diagnostic/prepare.py`由冻结burst源码生成独立Lab fork，
每stage首块设置计算时钟c=T0_s(min(M,12))；W13请求参考时间q=min(c,tau)，W2仍q=c。
首块计算与Q字节请求并行，首块完成取两者较晚时刻，之后执行原T0_s(M)-c私有增量。
请求结束即退出共享资源，计算尚未结束时保持team占用；W13完成后按冻结gap进入W2。
这里c是冻结首块时间的建模时钟，不是独立测得的纯计算下限；tau同样是有效服务窗口假设。
完整阶段无竞争严格保持T0。W13 alpha允许与冻结W2 alpha=0.08分离，共享容量仍统一。

先运行`overlap_diagnostic/check.py`：40个隔离形状/窗口组合保持基线，100个随机错峰混合
cohort在tau=inf且不覆盖alpha时复现旧模型（事件端点容差1e-7us），以及5个同步M4请求
在C=200、tau=100、alpha=0时W13结束精确等于5*8MiB/C。检查通过后才拟合。
没有改production/native或现有模型接口，候选退回边界是整个独立Lab目录。

`fit.py`固定T0、W2 alpha和gap，只用M4/M13 session1的n1/n2/n4、local/cross拟合；
目标是每条件内所有活跃expert W13的平均绝对误差，再等权平均条件，避免只优化前台。
粗网格C160..280步20、tau80..116步6、W13 alpha0..0.6步0.1，先按各worker-team
初始W13中位时刻排候选，前16名再按每条件全部31调用评分。训练最优C200/tau116/alpha0.2，
all-expert训练MAE4.292us。由于tau处于边界，扩展tau到80..164步12，其余协议不变，
`fit_expanded.py`得到C260/tau140/alpha0.5，训练MAE3.349us。
这些C都是各自模型的有效容量，不能解释成实测硬件带宽。扩展版改变统一C，所以W2虽未改alpha，
服务约束仍会变化，不声称只改变了W13时序。

下表为有竞争前台W13的逐条件配对误差MAE，含各形状两会话；M6/8/10/12/48未用于本轮拟合，
但都已观察过，故仅称开发迁移，不称新的独立验收。所有模拟以实测初始W13入口为条件。

| M | 原模型/us | 初始重叠网格/us | 扩展网格最优/us |
|---|---:|---:|---:|
| 4（含训练） | 15.405 | 6.533 | 9.331 |
| 6 | 15.377 | 6.652 | 9.363 |
| 8 | 11.792 | 8.717 | 8.945 |
| 10 | 2.790 | 5.053 | 10.524 |
| 12 | 2.589 | 3.698 | 2.169 |
| 13（含训练） | 1.605 | 4.227 | 2.133 |
| 48 | 5.452 | 6.046 | 5.110 |

扩展版M4/M6/M8 W2 MAE为7.206/7.015/6.810us，原为4.088/3.087/2.382us，明显回退。
降低训练误差并没有改善整体迁移，因此不采用该共享请求窗口重叠候选，也不在当前网格上
继续按M10验证残差挑选参数。这个负结果只否定本轮结构/校准组合，不能否定所有计算访存重叠模型。
需要独立识别kernel/阶段的请求服务与可隐藏部分，或用明确的kernel类别响应约束；不能把已有
六条经验历史净增量曲线当成已测的物理计算/访存分解。

完整粗网格、31-call重评候选、逐expert双GEMM残差和范围说明分别保存在
`overlap_diagnostic/results.json`与`expanded_results.json`，源身份在source_identity.json。
Ruff及定向静态diff检查用于本轮Lab/文档，默认planner保持原基线。大M、混合宽度、
真实完整计划和选择损失仍是完整目标的未完成项，不能用上述局部检查替代。

## 原生kernel结构核对与分阶段响应候选

只读核对`csrc/moe/arm/sve_bf16/jit_kernels.cpp`：构造器row_pairs=(rows+1)/2，
rows<=8时physical_rows=8、accumulator_base=16，否则physical_rows=12、accumulator_base=8。
`generate()`在BF16路径按physical_rows调用small_double_buffered_k_loop或m12_k_loop；
前者交替A/B寄存器bank，后者在m12_k4中分组加载/计算。W13存储/SiLU路径也随physical_rows分支。
因此8/9边界有独立源码依据，不是从误差表任意选择阈值。但不能仅凭源码断定它导致了全部竞争差异。
rows1..12是exact-M入口，存在六个row-pair计算量类别；奇偶尾行的存储谓词仍不同，
不能把“六类”解释成完全相同的六个完整kernel。后续特征至少区分row_pairs、K循环类别、stage与尾行。

本地和远端JIT源SHA一致：1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629；
远端扩展保持原dd554ea...。未改native代码或新增probe。既有eight_stage_demand PMU只覆盖
M1/12/13/48、独立stage且32份/1份B状态，不能直接填入本轮真实路径M7/M9首块服务参数。
现有BOnly/MatrixOnly探针也不完整覆盖所有行数，不伪称已获得各kernel的独立计算/访存分解。

新Lab候选`kernel_family_candidate/family_response.py`保留原比例资源模拟，仅对首窗口
min(M,12)<=8的实际small-loop类别使用
`alpha_s=0.08+beta_s*max(0,T0_s(12)/T0_s(min(M,12))-1)`，large-loop保持0.08。
W13 beta=0.632沿用M4第一场拟合；固定W13后，仅以同M4/session1/n1,n2/local,cross
条件的W2残差拟合beta_W2=0.458，训练RMSE0.618us。T0、C200、gap0.39和后台M>=12均不变。
该函数目前针对首请求窗口，不意味着后续尾块的资源暴露已完整建模。

已有M6数据的条件迁移：n1/n2/n4 W2偏差原为-1.797/-4.460/+3.004us，新为
+0.892/-0.210/+2.863us；W13仍与前一密度候选一致，n4有-5.828us残差。
M10全部条件逐字段恢复原模型。此为开发诊断，记录`kernel_family_candidate/transfer.json`，
不能算新鲜验收。60个M1..60隔离形状保持原输出；八组M9/10/11/12/13/24/48/60的
5-team错峰cohort保持原输出，验证不改变large-loop结果。Ruff通过。

准备命令`.venv/bin/python tmp/joint_cost_model_20260911/prepare_family_small.py`。
新M7/M9各8条件+anchor，两场seeds611561/611562、611581/611582；实际route嵌套前缀、
背景M48/24/12/12及0/1/2/4背景、local/cross、NUMA3/80workers/8T几何和原5warmup31pair协议不变。
完整profile、stage响应源快照、两种模型场景预测先于采集冻结于`family_small_m`。
M7 local_n4预测W13原167.385us、新188.553us；M9两版均188.504us，明确检验边界两侧。
远端运行`bash tmp/joint_cost_model_20260911/family_small_m/run.sh`已启动，四场串行；
启动前确认无重叠benchmark。当前采集中，未评分，不宣布通过或采用。默认planner仍不变。

## M7/M9前瞻结果与连续队列迁移

`family_small_m/run.sh`四会话全部COMPLETE，各369调用、共1476调用通过数值、实际M、
完整trace/isolation校验；session/compact本地保存，原始trace留在Arm-codex-internal同目录。
执行`family_small_m/evaluate.py`核对测前预测、source/frontier身份、seed和31个正式pair；
beta_W13=0.632/beta_W2=0.458未改，未用新数据重拟合。

有竞争为每M的local/cross×1/2/4背景×两会话共12条件：

| M | 指标 | 原比例模型 | kernel类别分阶段候选 |
|---|---|---:|---:|
| 7 | W13 MAE/us | 11.523 | 1.364 |
| 7 | W13 MAPE/% | 7.020 | 0.921 |
| 7 | W2 MAE/us | 2.369 | 0.929 |
| 7 | W2 MAPE/% | 3.570 | 1.359 |
| 7 | 前台完成MAPE/% | 7.248 | 3.891 |
| 7 | cohort完成MAPE/% | 1.924 | 2.158 |
| 9 | W13 MAE/us | 3.018 | 3.018 |
| 9 | W2 MAE/us | 0.799 | 0.799 |
| 9 | 前台完成MAPE/% | 5.215 | 5.215 |
| 9 | cohort完成MAPE/% | 2.076 | 2.076 |

M7双GEMM均改善，M9不再受到旧密度候选的额外修正，支持把实际K循环类别作为特征保留。
但新前台完成两组仍未通过MAPE<=3%、P90<=5%：M7 P90=7.129%，M9=8.212%。
M7/M9零背景双GEMM MAPE均<0.8%，零背景完成MAPE5.222%/6.063%，入口预测仍独立不足。
M7 cohort略回退，不只报告前台阶段的有利结果；上述并非完整planner验收。

`family_small_m/conditional.py`只代入实测初始W13时刻，不改冻结评分。M7 n1/n2/n4
W13偏差从-3.504/-10.784/-22.956us变为+1.314/-0.348/-2.447us，W2变为
+0.472/-1.351/+1.137us。M9两模型相同，n4 W13条件低估仍有7.936us，不能宣称large-loop响应无误。

为检查非首波任务，新增独立`family_dag_diagnostic` adapter，在原stage模拟器传递prepare和
on_complete回调；gather、publication与历史入口场景不变。24个DAG/publication一致性检查通过，
覆盖共享worker链、未模拟successor占用及零修正退化。这个adapter未进入默认planner。
`replay.py`读取旧queue_reduction的六种reduced图，每图15expert/五条8T lane、两会话，
新旧都使用publication模型，完整自主推演，不输入实测起点；各预测场景检查阶段/依赖顺序。
这是已观察数据的回顾性动态迁移，非新鲜验证、非完整224-expert计算。

队列完成总体MAPE0.966%→0.883%，但两个M8 expert的W13总体MAE8.709→8.976us，
W2 2.802→3.320us；平均完成分数会掩盖阶段回退。按顺序看M8 W13 MAE：

| 顺序 | 原publication模型/us | family候选/us |
|---|---:|---:|
| reverse（M8首波） | 15.153 | 2.572 |
| joins_reverse（M8首波） | 15.805 | 2.118 |
| forward（M8后续） | 5.648 | 15.842 |
| joins（M8后续） | 4.762 | 22.465 |
| staggered | 4.763 | 4.736 |
| serial | 6.125 | 6.125 |

首波改善不等于对后续环境通用；这个对照同时改变了背景形状、阶段与到达时刻，不能仅凭它
认定B历史是唯一原因，也不能给“后续M8”加反向常量。保留此反例，下一步分开诊断动态重叠误差
与后续环境敏感度，并继续扩展实际计划覆盖；不让单个微秒阶段的拟合替代最终选择损失验收。
所有冻结与诊断分别保存在family_small_m/evaluation.json、conditional.json和
family_dag_diagnostic/results.json。当前远端会话均已终止，无新增kernel/production变更，默认不切换。

## 后续M8回退的两层诊断：起点误差之外还有执行上下文差异

本轮只做已有queue_reduction数据的离线诊断，参数和生产路径未改，无新远端采集。
`family_dag_diagnostic/conditioned_starts.py`用每个任务实际W13起点替代入口/gather/通知预测，
但仍取max(实际起点,预测父任务完成)保持依赖合法。W13/W2及其后续重叠仍自主计算。
记录每次起点被推迟的clamp，不能把有clamp的结果称为完整实测时间线oracle。

M8 W13 MAE（两个expert×两会话条件汇总）：family在forward由自主15.842us降至
条件8.634us；joins为自主22.465us、条件23.800us。起点替换不能消除全部后续回退。
joins的930次job中248次被clamp、最大29.267us，故进一步构造下面无前驱预测的同波对照。
完整各计划、clamp及阶段残差见conditioned_starts.json。

发现reduced_joins最终波与reduced_joins_reverse第一波是同一组expert：
12/M8/CPU280–287、36/M8/288–295、81/M12/296–303、70/M12/304–311、24/M16/312–319。
脚本`matched_wave.py`逐字段断言expert/M/width/core_begin相同，仅模拟这五个job并代入各自
实际初始W13时刻。所有被排除任务逐样本检查：要么在该波首个W13前已结束，要么在该波
全部实际W13结束后才开始；两种预测的W13结束也早于下一批未模拟任务开始。
因此该对照没有预测前驱clamp，也没有遗漏同时执行的其他已记录任务。
它仍改变了前序两波工作/缓存/输出等上下文，不是仅改变B历史的独立因果实验。

同pair的“后续波W13减第一波W13”，负数表示后续更快：

| expert/M | session1差/us | session2差/us | pair MAD s1/s2/us |
|---|---:|---:|---:|
| 12/M8 | -20.260 | -20.030 | 2.420 / 3.320 |
| 36/M8 | -18.270 | -18.370 | 3.280 / 3.260 |
| 81/M12 | -11.480 | -14.970 | 3.260 / 2.090 |
| 70/M12 | -15.010 | -16.280 | 3.240 / 3.100 |
| 24/M16 | -20.150 | -24.720 | 2.910 / 5.370 |

这是五个任务共同的上下文差异，不只是M8局部修正失效。两个M8的实测条件中位数（两场平均）
第一波194.240/195.515us，后续波173.270/176.390us；family预测分别197.849/197.225us与
196.481/196.974us，只预测约0.3..1.4us变化。波内实际起点spread从11.070us变为7.390us，
已作为模型输入，不能把约18..20us实测差简单归到起点错位。
后续波family配对偏差仍+22.810/+20.344us，原响应为+5.247/+2.502us；模型确实无法用相同
无竞争成本和当前状态无关响应同时解释两种上下文。

下一步应保持该五expert波次不变，区分前序计算/访存、等待与准备状态对需求或服务的影响；
优先独立观测或可控上下文对照，不能按“第几波”直接扣常量，也不能断定是单一B热度或硬件频率。
这不撤销已测kernel类别差异，但说明可靠联合模型还需要能迁移的执行状态特征。
完整逐pair、几何断言、模拟结果与配对差保存在family_dag_diagnostic/matched_wave.json，
源码matched_wave.py；默认planner不变。完整计划、其他宽度与选择损失目标仍未完成。

## 固定目标波次的前序位置对照（已准备并启动）

E类Lab，复用既有queue桥接与transitive reduction，不改native/kernel/default模型。
准备命令`.venv/bin/python tmp/joint_cost_model_20260911/prepare_wave_context.py`。
目标固定expert12/36/81/70/24、M8/8/12/12/16、8T CPU280–319、full stripes；
五条件为first、one_same、one_cross、two_same、two_cross，加anchor共6计划。
一波前序使用原中间波M12/24/35/32/24；两波前序再加入原第一波M60/48/35/36/40。
同前序数量的same/cross任务、形状、相互依赖完全相同，仅前序核从CPU280–319移到CPU240–279。
目标核与窗口不变，目标等待全部前序完成，后续波等待目标全部完成，剩余原任务与依赖保留。
此对照区分局部worker/LLC与共享NUMA上下文，不是纯计算、纯访存或等时等待的因果分解。

静态检查全部任务共享核祖先关系、所有目标几何相等、前序全部先于目标与后续全部晚于目标。
先约简实验隔离冗余边，保留原图构造协议；不将约简视为生产图修改。
其他协议沿用NUMA3/80worker、H4096/F512 BF16 SVE256 Ntile16、W13/W2总8/4MiB、
owner1MiB/512KiB、4权重副本/216MiB scrub/固定workspace/early merge off、5warmup31pair。
seeds611601/611602，`wave_context/protocol.json`采前固定对照与配对MAD统计口径，不拟合新参数。
在确认无其他benchmark和原扩展SHA一致后，远端
`bash tmp/joint_cost_model_20260911/wave_context/run.sh`已启动两场串行采集。
当前采集中，无结果或通过声明。分析器wave_context/analyze.py将核对seed/frontier、数值/trace、
实际核与所有15个pilot依赖时序，再比较各目标W13/W2的逐pair差；源静态检查通过。

## 前序位置对照结果：局部执行环境贡献明显

wave_context两会话均COMPLETE，各246调用、共492调用通过数值/实际M/trace校验。
分析命令`.venv/bin/python tmp/joint_cost_model_20260911/wave_context/analyze.py`，进一步核对
frontier/seed、31个正式pair、实际core_begin及所有15个pilot依赖边。无参数拟合。
原始trace留Arm-codex-internal同目录，session/compact及analysis.json已本地保存。

下表为两个M8的逐条件配对差中位数再平均，负数表示有前序更快；括号内为平均pair MAD。

| 对照 | session1 W13差/us | session2 W13差/us |
|---|---:|---:|
| 1波同LLC - 无前序 | -21.775 (2.615) | -22.890 (2.290) |
| 1波跨LLC - 无前序 | -4.315 (3.600) | -7.045 (2.540) |
| 2波同LLC - 无前序 | -17.490 (2.345) | -19.955 (3.415) |
| 2波跨LLC - 无前序 | -6.960 (2.310) | -6.885 (2.405) |
| 1波跨LLC - 1波同LLC | +17.300 (2.825) | +15.655 (3.055) |
| 2波跨LLC - 2波同LLC | +10.400 (2.330) | +12.870 (2.535) |

所有差由原始pair独立计算，不要求不同配对中位数之间可相减。local/cross前序使用相同
expert、M、宽度与DAG，目标波次保持原expert/核/窗口；此差异在两会话方向一致且大于MAD。
目标波首W13中位起点：1波same约962.33/944.38us，cross975.61/965.85us；
2波same2189.40/2180.76us，cross2222.33/2190.06us。跨域等待并不更短，却没有获得同等阶段
加速，不支持“只要多等待一段时间就足以解释全部改善”的简单解释；仍未执行严格等时等待对照。

结果支持局部worker/LLC执行环境贡献，但当前same让目标所有核都参与过前序，不能区分
目标核自身状态、同LLC共享状态、代码/数据缓存或频率/执行单元状态，也不能排除共享NUMA的较小贡献。
不把cross差直接作为DRAM带宽系数，不增加“前序波数”常量修正。
下一独立对照应保持前序任务与调度相同，分别放到目标M8核、同LLC的其他核、另一LLC，
以区分目标核活动历史与域内共享状态；目标五expert并发仍保持不变。
本轮采集已结束，无远端任务运行；默认模型和planner未改变。完整模型/全计划/混合宽度目标仍未完成。

## 目标核与同域其他核对照（已启动）

新增E类Lab准备器`tmp/joint_cost_model_20260911/prepare_core_context.py`，复用wave_context
的同一中间前序波expert26/29/136/245/233、M12/24/35/32/24。统一改为两条8T链：
任务索引0→2→4与1→3，目标五expert全部等待这五个任务完成。前序执行位置为：
- target_cores：CPU280–295，即两个M8目标所用核；
- peer_cores：CPU296–311，即同LLC两个M12背景所用核；
- cross_cores：CPU248–263，另一LLC；
另保留无前序first和anchor，共5计划。目标五expert仍在CPU280–319、M8/8/12/12/16。
peer_cores会改变同波背景的活动历史，可能影响其请求强度，不称为纯LLC缓存干预。

三种前序计划逐字段验证只有前五个task_core_begins不同，其余bridge向量（含依赖）完全相同；
全部共享核任务对通过祖先检查，目标等待全部前序，后续工作仍被目标波依赖隔离。
此实验改变前序并行度但三种位置保持相同，不能直接与上一轮五lane前序当作单因素比较。

同原NUMA3/H4096/F512/BF16/SVE256/Ntile16、8T `(8,0,0,1,1)`、W13/W2 8/4MiB、
owner1MiB/512KiB、四副本/216MiB scrub/固定workspace/early merge off、5warmup31pair。
seeds611621/611622，协议与对照采前保存`core_context/protocol.json`，无模型拟合。
核实无其他采集及原扩展身份后，远端
`bash tmp/joint_cost_model_20260911/core_context/run.sh`已启动两场串行采集。
分析器逐expert报告W13/W2配对差与MAD，检查实际核与全部pilot依赖边；当前尚未评分。
默认planner、模型系数与native代码均不变。

## 两lane前序位置结果：目标核活动不是必要条件

core_context两会话均COMPLETE，各205调用、共410调用通过原数值/实际M/trace校验。
`core_context/analyze.py`核对采前frontier/seed、31正式pair、实际核及全部pilot依赖边；
原始trace留远端Arm-codex-internal同目录，session/compact和analysis.json本地保存。
没有修改任何cost系数或生产代码。

两个M8的逐expert配对差中位数再平均如下，负数表示前一条件更快：

| 对照 | session1 W13差/us | session2 W13差/us |
|---|---:|---:|
| 目标核前序 - 无前序 | -10.100 | -8.640 |
| 同LLC其他核前序 - 无前序 | -10.525 | -9.175 |
| 跨LLC前序 - 无前序 | -5.410 | -3.060 |
| 目标核前序 - 同LLC其他核前序 | -0.190 | 0.000 |
| 目标核前序 - 跨LLC前序 | -4.330 | -5.505 |
| 同LLC其他核前序 - 跨LLC前序 | -4.710 | -5.845 |

目标核减同域其他核的pair MAD为3.095/3.335us，未观察到目标核必须亲自执行前序的额外收益。
`intervals.py`用相同pair重采样索引同时处理两个expert，保留其相关性；4000次bootstrap、seed611623，
该对照95%描述区间为[-2.430,1.721]/[-1.875,2.395]us。没有预设等价界，不据此声称严格等价。
两种同域位置减跨域的区间分别为目标核[-6.175,-2.800]/[-7.255,-4.200]us、
同域其他核[-6.350,-1.605]/[-7.140,-4.132]us；这是逐对照描述区间，未作多重比较调整。

其他expert也保留了共同局部收益：M16的核CPU312–319在三种前序中都未执行前序工作，
但目标核/同域其他核前序后，其W13分别缩短10.14/11.08us、10.30/10.18us，跨域仅4.48/3.21us。
两个M12在同域前序后也普遍更快，逐expert明细与全部pair值保存在analysis.json。
结果支持共享域执行状态是必要研究方向，不能只给“这个team之前执行过任务”设置开关。
同域其他核前序也会改变背景状态/请求，当前仍不能唯一归因LLC数据缓存、共享时钟、预取或其他机制。

上一轮五lane前序的同域收益约18..23us，本轮两lane约9..11us；两组同时改变了前序并行度、
时长和近期活动分布，不能单独归为线程数或累计字节。下一步应对域级活动强度/时序与独立
服务观测做受控辨识，再接入有状态响应；不直接拟合“前序任务数/波数”补偿。
本轮运行已结束，默认planner未切换；完整计划、大M、混合宽度、选择损失仍需完成。

## 共享域历史特征辨识：累计工作量不足，峰值并发仍待新水平验证

本轮只读既有wave_context/core_context trace，`domain_history_diagnostic/analyze.py`在
目标波首W13之前提取同LLC已结束GEMM的分配核数×阶段区间。特征为前序expert数、逻辑
权重MiB、逻辑FLOPs、GEMM core-us、峰值分配核数，以及50/100/200/500/1000/2000us衰减
常数的EWMA活动。EWMA对每个区间[a,b]贡献w*(exp(-(t-b)/tau)-exp(-(t-a)/tau))。
这些都是时序/工作量代理，不是实际发请求核数、PMU字节或独立服务速率。

响应使用同pair两个M8的cross-minus-local W13差先平均，再取中位数，因此与早前
“每expert先取中位再平均”的汇总略有差异；原报告与原始数据不改。
同五expert前序（60MiB逻辑权重、1.598GFLOP）：五lane峰值40核局部收益
17.115/15.010us，两lane峰值16核仅3.815..5.775us。它们逻辑工作量相同，core-us也接近
（约21,903..22,018对20,406..20,508），所以累计任务/字节/FLOP/core-us不能单独解释差异。
两波五lane前序增至10expert、120MiB、4.354GFLOP、约58,000core-us，却只有10.485/12.660us
局部收益，也不支持累计工作越多就同比增强的简单规则。

仅用one_same/session1一个条件确定每种特征的过原点比例系数，全部其他已观察条件作
开发迁移（7条件，非独立验收），局部收益MAE如下：任务数/逻辑字节13.674us，FLOP17.216us，
core-us16.212us，目标前经过时长22.040us，峰值核数2.916us；EWMA按上述六时间尺度分别
6.229/4.735/3.972/5.662/8.652/11.536us。全部候选与残差保存，未按结果选一个进入模型。
只有16/40两个并发水平，而且两波与一波还改变历史分布，不能据此宣布峰值就是物理状态，
也不能把最优2.916us当新鲜验证。下一次若测中间24/32核，应事先冻结各候选，而非继续调整比例。

只读native fixed-team分支（fused_moe_bf16_tiled.cpp约9106–9157）：等待worker循环检查
completed_tasks、所属task_states/deps_remaining，无可运行任务时调用std::this_thread::yield。
所以未分配GEMM的worker并非自动零负载；trace的GEMM活动特征没有包含这种等待/调度流量。
源码未在此分支显示随等待时间指数退避，不能凭elapsed time假定轮询自动消失。
这只是待测的潜在干扰项，未证明它造成了上述域内差异，未修改等待实现。

完整逐pair特征、条件汇总与全部比例候选在domain_history_diagnostic/results.json。
本轮无远端采集、无模型参数或默认planner修改。独立共享服务证据与全计划泛化仍未完成。

## 24/32核前序新水平验证（测前冻结并启动）

`prepare_mid_context.py`复用同五个前序expert与目标五expert，前序改为3/4条8T链，
分别放在同LLC与跨LLC；加first、anchor共6计划。同lane数的same/cross bridge仅前五个
core_begin不同，全部任务资源祖先关系、目标等待全部前序均验证。目标M8/8/12/12/16与
CPU280–319、窗口、输入不变。其余NUMA3/80worker/四副本/scrub/workspace/early_merge off
及5warmup31pair协议沿用，seeds611641/611642。

仅用旧wave_context one_same/session1的cross-minus-same差标定：峰值系数0.427875us/core，
24/32核局部收益测前预测10.269/13.692us；累计任务对照两者均17.115us。
主统计是逐pair先平均两个M8差，再取31pair中位，与标定完全一致；不是以前部分表格的
先逐expert中位再平均。原报告不改。frontier/protocol/frozen_predictions.json采集前保存于mid_context。
峰值是候选活动代理，不是实测带宽；此实验不预测全计划完成，不自动触发模型采用。

远端确认无重叠benchmark和原扩展身份后，执行
`bash tmp/joint_cost_model_20260911/mid_context/run.sh`，两会话串行采集中。
分析器analyze.py检查数值/seed/frontier/实际核/依赖与逐pair差；score.py检查每调用实际
GEMM峰值等于计划24/32核，固定系数评分，4000次paired bootstrap仅描述测量区间，
不包含旧标定系数不确定度。当前未读取新时间评分，无新cost参数或默认planner修改。

## 24/32核前瞻结果：峰值代理优于累计任务，但仍系统高估局部收益

mid_context两会话均COMPLETE，各246调用、共492调用通过数值/实际M/trace验证。
analyze.py核对身份、31pair与全部pilot依赖；score.py确认每调用实际GEMM峰值均为计划24/32核，
没有用新实测峰值替换预测输入或重新标定。执行命令：
`.venv/bin/python tmp/joint_cost_model_20260911/mid_context/analyze.py`及同目录score.py。
原始trace留Arm-codex-internal同目录，session/compact及evaluation.json本地保存。

| 前序峰值 | 会话 | 实测局部收益/us | 峰值规则预测/us | 预测减实测/us | pair MAD/us |
|---|---:|---:|---:|---:|---:|
| 24核 | 1 | 7.550 | 10.269 | +2.719 | 2.290 |
| 24核 | 2 | 8.865 | 10.269 | +1.404 | 2.375 |
| 32核 | 1 | 10.585 | 13.692 | +3.107 | 3.270 |
| 32核 | 2 | 9.985 | 13.692 | +3.707 | 2.470 |

冻结峰值规则四条件MAE2.734us，累计任务规则（均预测17.115us）MAE7.869us。
峰值特征在新并发水平比累计任务数更有迁移价值，但全部四点高估，不能称为无偏精确模型。
实际收益bootstrap95%描述区间24核为[6.350,9.680]/[7.340,10.200]us，32核为
[8.580,12.915]/[7.515,10.600]us；这些只含新测量不确定度，不包含旧标定系数的误差。
原系数和预测保持冻结；本次不通过重拟合斜率或挑选会话消除偏差。

保留峰值/活动强度作为共享域状态候选，而非直接按过去峰值给每个任务减时间。
还缺少状态保持/衰减、连续队列中的作用位置、其他M/宽度，以及独立服务或请求观测。
下一步应将该状态假设与资源响应连接后验证整体计划和选择损失，不能用单个上下文差值
误差较小替代完整模型验收。当前无远端采集运行，默认planner及生产接口未改。

## 共享域状态连接资源响应：全stage服务变化与W13压力变化对照

本轮无新远端采集，只用已观察wave_context/core_context/mid_context作离线诊断。
`state_service_diagnostic/analyze.py`保留T0、kernel类别beta_W13=0.632/beta_W2=0.458、
gap和原请求窗口。对于同域前序，令C_eff=200*(1+gamma*P/40)，P为前序峰值16/24/32/40；
跨域对照的目标域取P=0。只用one_same/session1两个M8的local-versus-cross W13差拟合gamma，
网格0..0.5步0.001，得到0.088，逐pair差残差中位0.086us。目标初始W13起点来自实测，
因此不包含自主入口预测，也没有实现连续状态演化或衰减。
该C是有效代理，不是硬件带宽；没有直接从每任务时间减固定us。

统一服务变化使W13误差下降，但M12/W2回退。为区分作用阶段，固定同一个gamma=0.088，
另一个候选仅令W13的有效请求量Q_eff=Q/(1+gamma*P/40)，W2请求量与全局C=200保持。
`analyze_w13_scale.py`没有重新拟合；private成本和阶段基线完全不改。有效请求量是模型归一化，
不是新测得的DRAM流量或缓存命中率。所有stage统一归一化与增加C在当前公式内代数等价，
所以只凭该类时间拟合不能区分“容量变大”和“请求变少”的物理解释。

下表为12个同域条件会话（含唯一训练条件）的逐expert阶段配对误差MAE；M8/M12各两expert，
M16一个。全部是既有数据的条件分析，不称为独立验收：

| 分组 | 原family无状态/us | 全stage服务率变化/us | 仅W13压力变化/us |
|---|---:|---:|---:|
| M8 W13 | 19.398 | 7.510 | 7.543 |
| M12 W13 | 27.695 | 16.078 | 16.697 |
| M16 W13 | 16.521 | 4.813 | 5.553 |
| M8 W2 | 2.799 | 1.580 | 2.322 |
| M12 W2 | 1.432 | 3.252 | 1.429 |
| M16 W2 | 1.728 | 1.728 | 1.728 |

W13特定压力项保留大部分W13改善且避免M12/W2回退，但M12/W13仍系统高估约16.7us，
不能视为整体误差闭合。两波同域的局部收益也仍高估约4.5..6.8us，P相同并不代表状态相同；
状态演化/饱和/衰减尚待独立实验，而非把固定峰值永久带到后续任务。
24/32核已有数据的增量误差仍约1.7..3.5us，此次不是新的测前验证，不覆盖跨域并发任务的
状态分配，也不据此切换planner。

`check.py`验证60个隔离M保持原基线、100个随机混合cohort的统一Q缩放/C变化等价性
（端点容差1e-7us），以及5个非法scale拒绝，全部通过；Ruff与静态diff检查通过。
独立Lab源scaled_stage.py/family_scaled.py和两份完整残差保存于state_service_diagnostic。
默认模型/生产kernel未改，真实全计划、混合宽度、大M和选择损失仍属于未完成目标。

## 自主DAG状态候选：后续M8改善，完成时间总体基本持平

M/E类独立Lab原型保存在dynamic_state_candidate，未修改默认分析器、planner或native。
新增实际activation时刻回调：当预测W13真正开始时才读取状态，不能在构造future job或
计算未来gather起点时提前读取。每个LLC维护running expert核数、曾经峰值和完成事件锁存峰值。
W2完成时锁存该域峰值；之后启动的W13取Q_scale=1/(1+0.088*latched_peak/40)。
本轮“完成事件锁存且本call内不衰减”是明确的最小状态演化假设，不是已经辨识的硬件规律。
状态每次预测调用重置，不跨请求继承；只支持NUMA3内不跨LLC的8T team。
active expert区间从W13到W2完成，含阶段间隙，不声称等同实际发请求活动。

使用原family beta0.632/0.458、C200、T0、gather/历史入口场景和publication参数；gamma=0.088
沿用旧上下文标定，没有用queue数据拟合。请求速率变化参与后续事件推进与重叠，W2请求量
和无竞争时钟保持；不从任务完成时间扣固定us。prepare/on_complete继续负责原依赖、通知和worker复用。

先验证30个DAG零gamma退化案例、延后root应看到准备之后/激活之前的完成事件、跨域隔离、
同波初始任务不会因当前启动而自动获得历史状态、跨调用状态重置。全部通过，Ruff通过。
检查命令`.venv/bin/python tmp/joint_cost_model_20260911/dynamic_state_candidate/check.py`。
identity.json记录原型与冻结profile身份。

`replay.py`完整自主推演旧queue_reduction六种reduced图，15expert/五8T lane、两会话；
只输入计划与历史31个相关入口场景，无当前实测起点，未重新拟合。它是回顾性动态迁移，
非新鲜验收、非完整224-expert计划。各预测场景再次检查阶段与依赖顺序。

| 指标 | 无状态family | 自主状态候选 |
|---|---:|---:|
| 两个M8 W13 MAE/us | 8.976 | 4.262 |
| 两个M8 W2 MAE/us | 3.320 | 2.946 |
| 队列完成MAE/us | 27.777 | 28.053 |
| 队列完成MAPE/% | 0.883 | 0.921 |

完成时间总体基本持平，没有证据据此宣称选择效果变好；较小平均分数差也不应放大成明显退化。
后续M8阶段改善更明确：forward W13 MAE15.842→4.971us、joins22.465→5.625us，
首波reverse/joins_reverse保持2.572/2.118us。串行W13仍6.125us，状态归一化不会改无竞争T0，
不能掩盖该基线/上下文残差。forward完成MAE23.516→31.897us、joins13.119→18.680us回退，
joins_reverse29.358→9.825us改善，其余详细数值保留，不能只选有利顺序。

median-ready诊断路径中，首波M8锁存值0/scale1，forward/joins后续M8锁存40/scale0.919118，
serial后续M8锁存8/scale0.982704；这些状态由预测事件生成，不是按plan名称设置。
结果与路径分别保存在results.json的predictions/records/summary和state_paths_for_median_ready。
该路径只用于检查状态，评分仍使用完整31场景，不把median-ready结果替代冻结场景评分。

保留自主状态候选做后续新队列/更大覆盖的验证，但尚未证明状态衰减、跨LLC team、混合宽度、
大M、完整计划或搜索选择损失。默认planner仍不切换；本轮无远端采集、无提交/推送。

## 自主状态模型的新顺序与固定候选选择验证（已冻结并启动）

`prepare_state_order_holdout.py`使用seed611680，在相同五条8T lane、每lane原三个expert内
独立打乱顺序，排除本joint-model现存frontier里的队列组合，生成10个新组合；另加forward/
reverse控制和anchor，合计13计划。原15expert/M/核/窗口/任务集合不变，图先约简并验证共享核祖先。
只检验执行顺序，不是新team组合或完整224expert搜索空间。

所有预测在采集前冻结于state_order_holdout：publication、family、dynamic三模型共享旧T0、
gather/相关入口场景与publication参数；dynamic gamma0.088与不衰减锁存规则不变。
每模型按预测完成中位数直接取最小值，12候选中均选new_06，预测分别1990.430/1989.812/1988.972us。
候选没有因同赢家而重选，也不额外搜索有利于新模型的集合。本批可以检验共同选择相对集合内
实测最优的损失，不能用相同选择声称某模型选择更好。所有数值场景、模型源快照和winner采前保存。

目标计算时间仍从scheduled_compute到15pilot最后W2结束，最终merge和剩余expert不属于该指标。
新顺序误差单独统计10×2条件，forward/reverse控制不混入新鲜形状/顺序的评分；选择oracle为
全部12候选的每会话实测中位数最小值，并报告所选减最优的逐pair差/MAD。
两会话只是同一工作负载的重复，不能把两点选择损失当成跨工作负载的regret分布。

原Arm-codex-internal NUMA3、80workers、8T full stripes、H4096/F512、BF16/SVE256/Ntile16、
4权重副本/216MiB scrub/固定workspace/early_merge off、5warmup31pair协议不变，
seeds611681/611682。确认无其他benchmark与原扩展身份后，执行
`bash tmp/joint_cost_model_20260911/state_order_holdout/run.sh`，两场串行采集中。
分析器evaluate.py检查source/frontier/seed、实际核和所有pilot依赖，读取冻结预测评分，
不输入新起点重新模拟，不拟合新系数。当前尚未评分，默认planner与生产代码未变。

启动修复记录：首次run.sh在采集前以exit1结束，路径替换误将queues_runner变成不存在的
state_order_holdout_runner；没有生成session1.json或测量数据。保留原run.sh，新增run_fixed.sh
只修复runner路径并另写runtime_identity_v2.txt；准备器改用整行r=匹配，避免再次修改runner变量。
冻结frontier、模型源、全部预测、seed与winner完全不变。确认旧进程已终止、目标会话输出不存在后，
已启动`bash tmp/joint_cost_model_20260911/state_order_holdout/run_fixed.sh`，当前仍在采集。

## 完整原计划支持域审计

`coverage_audit/analyze.py`以weights_only读取原route文件，核对route SHA、layer4、expert13
原M60未变化，并断言anchor任务expert集合与实际route计数集合一致。原计划224expert、
12288条路由，最大M1718。这里统计输入路由行，不将其解释为实测运行时间份额。

| 原anchor类别 | expert数 | 路由行数 |
|---|---:|---:|
| 8T、M<=60（当前接口支持域） | 146 | 2124 |
| 8T、M>60 | 36 | 7709 |
| 16T、M<=60 | 39 | 587 |
| 16T、M>60 | 3 | 1868 |

当前8T模型接口可覆盖约17.3%路由行，不等于这些任务的完整联合精度都已验证。
现有15expert受控前缀共402条路由，约3.3%，其核放置与原anchor部分不同，不能把
受控前缀的选择损失当成原计划整体选择效果。8T大M约62.7%路由行，16T合计约20.0%。
大8T实际形状包括M62..307、376/529/714/768/1205；16T大形状为73/77/1718。
因此本轮新顺序评分之后应优先补大M8T与16T基线/联合覆盖，保留后续1/2/4T和完整搜索目标，
不能继续用小前缀阶段改善代替主工作量验证。完整逐expert形状/宽度/核/窗口清单在
coverage_audit/results.json，本轮只读审计，没有外推或自动采用fallback。

## 新队列顺序前瞻评分：完成误差约0.48%，固定集合选择接近最优

run_fixed.sh两会话均COMPLETE，各533调用、共1066调用通过数值/实际M/完整trace校验。
原始trace保留Arm-codex-internal同目录，session/compact本地已取回。
`state_order_holdout/evaluate.py`核对全部冻结模型源、frontier、seed、31个正式pair、实际核
和所有pilot依赖边；评分仅读测前预测，无新起点输入或重拟合。

10个新顺序×两会话共20条件的15-expert计算完成结果（控制计划不混入新顺序误差）：

| 模型 | 完成MAE/us | MAPE/% | P90绝对相对误差/% |
|---|---:|---:|---:|
| publication | 9.359 | 0.468 | 0.911 |
| family | 9.373 | 0.468 | 0.733 |
| dynamic | 9.603 | 0.479 | 0.777 |

三者均通过本批MAPE<=3%、P90<=5%门槛，完成精度接近，没有证据认定dynamic更准确。
但阶段结构有独立收益：新顺序首波M8 W13/W2 MAE由publication的17.293/4.107us降至
family/dynamic的2.942/1.065us；后续M8 W13/W2由family6.233/1.829us降至dynamic4.368/1.699us。
此阶段改善没有转化为不同选择，三者测前均选择new_06：

| 会话 | 所选new_06实测/us | 集合内中位数最优 | 最优实测/us | 所选损失/% |
|---|---:|---|---:|---:|
| 1 | 1986.220 | new_04 | 1981.060 | 0.260 |
| 2 | 2004.460 | new_01 | 2002.830 | 0.081 |

直接配对所选减当场最优的差中位数为-6.390/-0.320us，MAD27.900/20.530us。
这与“各自中位数之差”不是同一统计，符号不必相同；保留两者，不能强行调成一致。
从波动看这些计划接近并列，不能宣称实测oracle稳定优于所选或新模型比旧模型选得更快。
两次集合内损失很小，但只是同一工作负载的两次重复，不是广泛regret分布验收。

本批支持继续保留8T自主状态候选：新顺序完成门槛通过，首波/后续M8阶段迁移改善。
其范围仍限15expert受控前缀。覆盖审计表明原计划主要剩余工作是大M8T与16T，应优先扩展
这些基线与需求/联合验证，再进入完整计划及相同预算搜索。没有把此次局部通过标记为完整目标完成，
默认planner未切换；所有远端会话已终止，无提交/推送。

## 大M8T真实路径基线扩展（训练采集中，留出未启动）

针对覆盖审计中的主要缺口，新增prepare_large8_baseline.py，使用原route layer4的
expert85（原M1205），同一packed B与目标输入前缀，固定8T CPU312–319、full stripes
`(8,0,0,1,1)`、Ntile16、W13/W2总8/4MiB、owner1MiB/512KiB。
12训练形状：12/60/72/96/144/240/384/600/840/1080/1200/1205；
12留出：62/95/157/234/283/376/529/714/768/961/1093/1199，两个集合不重合。
每批加原anchor，13计划。移除的目标route转给已有expert，保持top-k唯一性和active集合；
其他expert全部等目标完成，不能描述成整个输入计数完全不变。24个嵌套前缀位置与active集合
已本地验证；isolate_bridge保留原非目标依赖，所有共享核任务对静态祖先检查通过。

NUMA3/80workers、四权重副本/216MiB scrub/固定workspace/early_merge off、5warmup31pair
沿用。训练seeds611701/611702，留出预留611721/611722。确认无重叠benchmark及原扩展身份后，
远端`bash tmp/joint_cost_model_20260911/large8_baseline/train/run.sh`启动两场串行采集，
当前第一场已校验通过；validation/run.sh没有启动。所有前沿/协议保存large8_baseline。

在读取训练时间前，准备fit.py定义主候选：M<=60完全保留旧模型；对M=12b+r>60，
完整前缀为T0_s(60)+d_s*(b-5)，每stage一个d_s只用session1的M>60且r=0点拟合。
尾块使用原独立row-pair历史曲线h4的值（含偶数行offset），假设后续历史保持该平台，
不按M12计算量比例缩放。M1205尾块用于诊断，不参与d_s拟合。此平台是假设，尚未验证；
M12/M60新目标权重控制用于检查旧锚点可迁移性，不能自动覆盖旧基线。

第二场只检查重复性。拟合后必须先保存模型和留出预测，再启动留出采集；若训练已否定
稳态斜率假设，应记录失败并在未读取留出数据前重定候选，不能用留出反复调参。
本轮是M/E Lab基线扩展，不含大M对外访存需求、联合减速或16T模型的通过声明，默认planner不变。

## 大M8T训练结果：完整块稳定，h4尾项平台不足

训练两会话均COMPLETE，各533调用、共1066调用通过数值/实际M/trace校验；原始trace留远端，
session/compact已本地取回。运行large8_baseline/fit.py，仅session1完整块点拟合，保存
model.json及原主候选的留出预测，但validation/run.sh仍未启动，尚无留出数据。
每12行稳态斜率W13=150.469037us、W2=72.293234us，保持原M60锚点和M<=60输出不变。
原主候选整阶段MAPE：训练第一场W13/W2为0.288%/0.677%，第二场重复为0.239%/0.554%。
所有训练/重复点最大相对误差W13为1.085%、W2为1.990%。这不是未见大M泛化结果。

同权重新目标M12/M60控制也支持旧锚点基本可迁移：W13预测161.180/777.050us，
实测第一场159.450/773.570us、第二场160.270/777.670us；W2预测82.485/396.160us，
实测82.350/395.800及82.750/395.090us。未因新expert身份重新改旧基线。

独立查看未用于稳态斜率拟合的M1205尾块诊断，不能被15ms大阶段的低百分比误差掩盖：

| M1205-M1200整阶段增量 | h4平台预测/us | session1配对实测/us (MAD) | session2/us (MAD) |
|---|---:|---:|---:|
| W13 | 105.995 | 71.340 (9.250) | 68.550 (11.550) |
| W2 | 36.919 | 18.570 (15.330) | 14.520 (9.950) |

这里是添加尾块后的整个GEMM净增量，包含可能的scratch/布局变化，不是独立panel服务时间。
旧h4平台在长历史r5上明显偏高，不能宣称尾块模型已泛化。原短历史曲线也不能直接外推：
W13 r5的旧拟合offset仅1.937us、decay0.0323，来自h1..3三点，offset并非可靠物理下限。
保留model.json作为已冻结但存在缺陷的主候选，不覆盖原记录、不启动留出调参。
下一步补长历史完整基点与各row-pair尾项的训练对照，明确尾项平台后再冻结修正版和采留出。
完整块斜率结果保留，不把此尾项缺陷解释成全部大M成本不准确；大M对外需求与联合响应仍未测。
配对明细和重现脚本为large8_baseline/tail_diagnostic.json及tail_diagnostic.py。
当前远端采集均结束，默认planner未改变。

## 长历史六类尾项补充训练（已启动）

prepare_large8_tails.py复用expert85/8T/CPU312–319的同一隔离桥接与原协议，分别以M600
（h50）和M1176（h98）为完整基点，追加r1/3/5/7/9/11，覆盖六个row-pair类别。
每组7形状+anchor，两场；seeds611741/611742与611761/611762。所有新增点与原12留出形状
不重合，原large8_baseline/model.json和失败h4平台诊断保留，不覆盖。
被移除route仍按原nested helper转给目标之后执行的已有expert，未声称全部输入计数不变。

采前确认原扩展身份和无重叠benchmark后，远端
`bash tmp/joint_cost_model_20260911/large8_tails/run.sh`按h50两场、h98两场顺序启动。
当前h50第一场已校验通过。普通页/NUMA3/80worker/4权重副本/216MiB scrub/固定workspace/
early_merge off、5warmup31pair、8T full stripes、Ntile16、owner1MiB/512KiB均不变。
没有修改native/kernel或默认planner，未启动原留出集。

large8_tails/analyze.py将逐pair计算T_s(base+r)-T_s(base)，保留中位数、MAD和全部原始差；
负数或高噪声净增量不截断为“零成本kernel”。这些仍是整阶段净差，可能包含scratch/layout影响，
不是直接panel计时。每个h的session1可用于尾项拟合，session2只作重复检查；h50/h98是否接近
平台由数据判断，不能预先把其中一个强行当作极限。源和shell语法静态检查通过。

## 长历史尾项结果：小尾项仍随历史变化，W2存在负净增量

large8_tails四会话均COMPLETE，各328调用、共1312调用通过数值/实际M/trace校验。
analyze.py核对frontier/seed和31个配对样本；原始trace保留Arm-codex-internal同目录，
session/compact与analysis.json本地保存。原12个留出形状仍未采集。

表中为两个会话各自配对中位数，单位us：

| 尾行r | W13 h50 | W13 h98 | W2 h50 | W2 h98 |
|---|---|---|---|---|
| 1 | 62.31 / 60.36 | 46.83 / 42.39 | -17.75 / -10.68 | -9.92 / -17.32 |
| 3 | 73.19 / 73.69 | 60.29 / 61.30 | -6.53 / 2.79 | 3.67 / 2.75 |
| 5 | 79.25 / 83.25 | 78.89 / 71.40 | 8.77 / 22.19 | 21.63 / 9.55 |
| 7 | 96.74 / 92.17 | 95.34 / 93.55 | 19.23 / 20.19 | 27.06 / 19.51 |
| 9 | 131.08 / 130.23 | 133.40 / 135.21 | 56.20 / 62.70 | 65.67 / 50.94 |
| 11 | 152.74 / 151.24 | 153.87 / 149.37 | 79.31 / 73.00 | 81.66 / 70.53 |

W13 r1/r3在h50到h98仍下降，不能把所有类强行设为同一个热平台。
W2 r1的负净增量四组同向；4000次paired bootstrap、seed611763的95%描述区间分别
[-23.33,-9.66]、[-24.37,-5.26]、[-14.41,-2.73]、[-27.64,-3.90]us，均不含0；
这些是逐对照描述区间，未作多重比较校正。其他小W2尾项噪声较大，全部MAD/原始差保留。
不能将负数截断成零或解释成尾kernel负执行时间。

worker_check.py进一步比较每CPU的W2区间：r1在h50两场分别7/8、6/8个worker的配对中位差
为负，h98两场均8/8为负；每worker差通常约-0.3..-6.5us，比team包络净差小。
因此不能仅用某一个worker或团队端点解释全部变化，也不能把team净差直接当作独立尾kernel成本。
当前exact-M dispatch源码先执行原完整M12块，再调用尾kernel；完整块数并未因r1而减少。
但W13结束状态、dispatch、scratch/layout和worker时序仍可能影响整个W2，具体因果尚未验证。

这批数据否定了“原完整前缀成本不变，再加一个非负、独立尾成本”的简单解释，未否定完整
GEMM成本可预测。下一版应把长历史项明确命名为有符号的形状/阶段净修正，保持最终阶段时间
正值与原M<=60基线，不把净修正用于计算物理请求量或kernel服务下限。先冻结修正版再采留出，
并检查是否需要补齐原留出中缺失的r8/r10类别；原冻结h4候选保留作对照。
当前无远端采集运行，默认planner未改变。

## 有符号形状修正已冻结，18形状留出启动

新增独立Lab large8_baseline/large_model.py。M<=60直接返回原模型；较大M=12b+r的完整前缀
保持原M60锚点及已冻结150.469/72.293us每12行斜率。r>0时，对每stage、六个odd row-pair类别
使用h4旧曲线值及h50/h98的session1配对净增量作三个节点，按history分段线性插值；偶数尾行
仍加原cold even-offset。仅在h99/100沿最后一段延伸，范围显式限M<=1205，不外推到更大M。
节点允许为负，名称和语义是“全阶段形状/历史净修正”，不是负kernel执行时间或负资源请求。
最终阶段值必须为正，并且不小于其首块参考时间；未提供M单调性或单调剪枝保证。

使用训练数据第一场节点，未用第二场更新参数；完整块斜率未重拟合。遍历M1..1205检查
正阶段时间与首块下界，M1..60逐字段与旧模型相等，全部通过。原model.json的h4平台候选保留。
model_v2_frozen.json为最终测前记录；格式化前源另存large_model_preformat.py，格式化后重算
18个预测逐字段相同，原model_v2.json也保留。评分器按声明的旧源归档与当前源分别核对SHA。

原12留出不变，补充104/106/608/610/1160/1162六点，覆盖原集合缺失的r8/r10，且与全部
训练点不重合。validation_extra为独立7计划批（含anchor），新seeds611781/611782；原留出
仍为611721/611722。全部18点预测在采集前冻结，没有按留出结果选择插值或节点。

原NUMA3/80worker/8T CPU312–319、H4096/F512、BF16/SVE256/Ntile16、full stripes、
owner1MiB/512KiB、4权重副本/216MiB scrub/固定workspace/early_merge off、5warmup31pair不变。
核对无重叠benchmark与原扩展后，远端
`bash tmp/joint_cost_model_20260911/large8_baseline/run_validation.sh`已启动原留出两场，
随后补充留出两场，严格串行。evaluate_validation.py仅读冻结预测，报告W13/W2及双GEMM合计
的绝对误差、MAPE/P90和逐形状噪声，同时保留h4对照；不把整阶段结果当作物理尾成本验证。
当前采集中，默认planner未改变，大M需求/联合和16T仍待后续完成。

## 大M8T独立留出结果：有符号形状修正通过本轮基线门槛

原12点留出两场各533调用，补充6点两场各287调用，共1640调用全部通过数值、实际M、
trace/isolation校验。evaluate_validation.py核对冻结模型源（含声明的格式化前归档）、
frontier/seed、31个正式pair，仅按采前预测评分，没有重新拟合。完整结果validation_results.json。

18形状×两场共36条件，范围M62..1199，包含完整块与各类奇偶尾形状：

| 模型 | 指标 | MAE/us | MAPE/% | P90绝对相对误差/% |
|---|---|---:|---:|---:|
| 原h4平台 | W13 | 13.326 | 0.203 | 0.414 |
| 有符号修正 | W13 | 7.868 | 0.183 | 0.400 |
| 原h4平台 | W2 | 16.167 | 0.714 | 1.549 |
| 有符号修正 | W2 | 6.484 | 0.394 | 1.091 |
| 原h4平台 | 双GEMM合计 | 26.624 | 0.270 | 0.473 |
| 有符号修正 | 双GEMM合计 | 11.846 | 0.153 | 0.265 |

修正版W13/W2最大相对误差0.834%/1.875%，双GEMM最大0.570%。两版本整阶段都通过
本批MAPE<=3%、P90<=5%门槛，修正版绝对误差与平均偏差明显缩小，但不能声称每点都改善。
合计是每pair的W13+W2，不包括gather、阶段间隙和初始入口，不能称为完整expert/e2e完成时间。
本轮支持保留large8_baseline/model_v2_frozen.json作为8T/M<=1205的无竞争阶段候选，
不把有符号净修正作为物理尾成本、负访存或单调剪枝依据；联合环境和对外需求仍未由此验证。
所有8T留出采集均已结束，原始trace留远端同目录，默认planner未切换。

## 16T首批标定已准备并启动

prepare_width16_baseline.py使用同route的expert96（原M1718），隔离到16T CPU304–319，
几何`(16,0,0,1,1)`、Ntile16、W13/W2全stage8/4MiB、owner512/256KiB。
两批各12形状+anchor：first_rows=M1..12，bulk=12/13/24/36/48/60/96/384/768/1200/1716/1718。
M12跨批作为控制；共23个不同嵌套前缀已验证target计数和active集合，桥接全资源祖先检查通过。
这是初始发现网格，尚未覆盖所有16T历史/尾项，不借用8T系数宣称16T模型已经建立。

沿用NUMA3/80worker、真实route、4权重副本/216MiB scrub/固定workspace/early_merge off、
5warmup31pair协议，seeds611801/611802、611821/611822。8T采集终止并确认无重叠任务后，
远端`bash tmp/joint_cost_model_20260911/width16_baseline/run.sh`已启动，四会话顺序执行。
所有源/协议/前沿在width16_baseline；当前未读取16T时间结果或拟合参数，默认与生产接口保持。

## 16T首批结果：重复性与完整块斜率，尾形状仍未闭合

width16_baseline四会话均COMPLETE，各533调用、共2132调用通过数值/实际M/trace校验，
analyze.py进一步核对frontier/seed和31正式pair。原始trace留Arm-codex-internal同目录，
session/compact及analysis.json本地保存。本轮不从8T参数缩放得到16T成本。

第一场预测第二场的同形状重复：W13 MAE2.026us/MAPE1.027%，W2 2.346us/1.030%，
gather 6.825us/8.699%。gather仍需分开检查worker到达与服务，不能吸收到GEMM斜率。
first_rows第一场M1..8 W13约76.12..80.63us、W2约33.93..36.42us；M12为91.21/45.65us。
这些是16T实际owner512/256KiB下的独立测量，不是8T时间除以2。

只用bulk/session1中M>60的完整12行倍数，保持该场M60锚点423.60/202.76us，得到
W13/W2每12行斜率75.245684/36.431865us。第二场完整块检查MAPE0.384%/0.709%，
最大相对误差0.984%/2.419%。这是同形状重复诊断，尚未建立任意尾形状的16T模型或做留出验收。

配对净增量显示短/长历史不能混用：M13-M12 W13两场68.64/69.28us、W2 26.16/22.85us；
M1718-M1716 W13仅7.90/11.14us（MAD12.19/7.98），W2 -7.45/-12.53us（MAD22.36/33.00）。
长历史小W2净增量噪声很大，不能从这两个点断定物理尾成本为负或精确估计热平台。
完整明细与原始pair差保存在analysis.json，不截断负值，也不宣布已有16T尾成本模型。

## 16T六类尾项训练启动

prepare_width16_tails.py准备h1/h2/h4/h140，即M12/24/48/1680完整基点各加r1/3/5/7/9/11。
每批7形状+anchor、两场，共八会话，seeds611841/611842、611861/611862、611881/611882、
611901/611902。沿用expert96原M1718嵌套前缀、16T CPU304–319、full stripes和原NUMA3/
80worker/4副本/216MiB scrub/固定workspace/early_merge off、5warmup31pair协议。
所有点都属于训练，session1拟合、session2重复，之后另行冻结留出；不借用8T尾项节点。

确认前批终止且无重叠benchmark、扩展身份一致后，远端
`bash tmp/joint_cost_model_20260911/width16_tails/run.sh`已启动，八会话严格顺序。
当前未读取新尾项时间或拟合，默认planner/生产kernel未改变；大M对外需求和完整联合仍待验证。

## 大M持续背景下延后小M的零竞争预测（已冻结，未启动）

等待16T尾项采集期间，prepare_steady_background.py准备了独立联合对照。目标expert12/M8
固定8T CPU312–319，先在同team执行expert37/M48；后台expert85/14/115/59分别为
M1205/768/714/529，8T CPU280–311，启用0/1/2/4个。其余后台等目标结束后才启动，所有
非cohort任务等六个所选expert全部结束；完整原route计数不变（nested配置仍保留原M60，未截断）。
前序与目标共享核有依赖，所有任务共享核祖先关系静态验证。inactive后台排在目标之后，
避免用反向依赖破坏拓扑顺序。

模型快照将已验证large8 baseline接到原dynamic DAG，其他参数/gather/入口/publication不变。
所有31个历史入口场景下，活跃后台的“预测首块请求结束”比目标W13启动至少早264.457us，
目标W2结束也早于后台W13结束；所以当前首块请求+其余private的结构严格预测：
四条件目标W13均125.440us、W2均59.750us，无竞争增量。此处private是模型状态，不是
已观测的真实panel缓存状态；未来必须检查实际后台阶段覆盖和已运行时长。

前沿、完整模型/基线/profile快照、零增量数值预测与源身份保存在steady_background，
预留seeds611921/611922，5计划含anchor、原5warmup31pair协议。该实验可检验首块后不再
产生请求的假设，但若出现差异，仍需区分稳态请求、共享域状态和其他影响，不能直接等同DRAM因果。
当前只准备，没有上传或启动，与正在运行的16T测量不并行。默认planner未改变。
16T尾项进度：h1两场、h2第一场已校验通过，其余继续按原脚本顺序运行。


## 16T尾项采集完成：长历史W2出现worker完成时间双峰

远端`bash tmp/joint_cost_model_20260911/width16_tails/run.sh`八场全部终止且COMPLETE。
沿用expert96/16T CPU304–319、NUMA3/80worker、H4096/F512 BF16/SVE256 Ntile16、
full stripes `(16,0,0,1,1)`、W13/W2总8/4MiB与owner512/256KiB、4副本/216MiB scrub、
固定workspace、early_merge off、5warmup31pair。`width16_tails/analyze.py`对八场核对
frontier身份、seed、数值与isolation标记和31pair，输出analysis.json；原始trace留远端同目录。

r1的配对整个阶段净增量：h1 W13两场69.55/69.97us，W2 24.42/25.59us；
h4 W13 58.95/53.31us，W2 8.96/10.04us；h140 W13 15.07/9.68us，
W2 -292.68/-28.69us。最后一项不具备可直接拟合成尾成本的重复性。

进一步逐worker检查保存在long_w2_worker_diagnostic.json。M1680 W2阶段中位时间
两场5421.79/5134.41us，但每次调用的worker时长中位数再取中位仅5115.185/5120.68us。
以事后描述性阈值max_worker-minus-median_worker>150us筛查，两场分别16/31和15/31次，
典型慢worker落后约300us；最后完成CPU分散，W2最大arrival spread仅1.38/1.94us。
因此整阶段中位数受两峰混合比例跨过一半影响，不能把约-293us解释为物理尾成本。
M1681/M1683/M1687两场均无上述宽完成差，M1689/M1691也出现该现象，说明与形状有关，
但当前没有识别其硬件/调度/内核因果。保留全部样本，不删除慢样本或用worker中位替代计划完成时间。
16T长期W2尾项拟合暂缓，需从原trace或后续对照区分服务时间与偶发慢worker；不宣称16T基线已验收。

## 大M持续背景对照已启动

前批exec句柄正常exit0，远端进程检查无旧benchmark，扩展SHA仍为
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`。
上传此前冻结的steady_background独立目录后，执行
`bash tmp/joint_cost_model_20260911/steady_background/run.sh`，两场seeds611921/611922串行。
沿用前节冻结的0/1/2/4大M后台、M48前驱后M8目标、8T实际full stripes、相同route与运行协议。
当前测量中，尚无实测竞争结论；冻结的首块后零压力模型仍预测所有条件M8 W13/W2为125.44/59.75us。
默认planner、kernel和生产接口未修改。后续先检验实际阶段覆盖，再比较配对增量。


## 大M W13覆盖延后M8：两场未发现大幅持续减速

`steady_background/run.sh`两场均COMPLETE/exit0，每场205调用、共410调用通过数值和trace/isolation。
`steady_background/analyze.py 1`与`... 2`核对冻结frontier、18份model源、seed、扩展身份和31正式pair，
只评采前预测；Ruff通过。输出analysis_session1.json/analysis_session2.json保留逐pair增量和覆盖。
原始trace留Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/joint_cost_model_20260911/steady_background。
配置仍为NUMA3/80worker、目标expert12/M8在CPU312–319的8T team，M48前驱后启动；
后台M1205/768/714/529在同LLC其余四个8T team，H4096/F512 BF16/SVE256 Ntile16、
`(8,0,0,1,1)`、stage8/4MiB、owner1MiB/512KiB、4副本/216MiB scrub、固定workspace、
early_merge off、5warmup31pair，seeds611921/611922。

| 后台数 | W13配对增量/us 两场 | W13配对MAD/us 两场 | W2配对增量/us 两场 |
|---|---:|---:|---:|
| 1 | 0.96 / 0.38 | 1.09 / 1.35 | -0.13 / -0.26 |
| 2 | 1.28 / 0.27 | 1.29 / 1.96 | 0.35 / -0.12 |
| 4 | 2.55 / 0.77 | 0.87 / 1.84 | 0.28 / 0.03 |

n0实测W13中位120.97/120.95us，W2 59.57/59.57us；冻结模型所有条件125.44/59.75us。
因此当前绝对误差中已有约4.5us的n0 W13高估，不应通过竞争参数消除。两场结果均未显示
该条件下的大幅持续减速，但没有做预声明等价检验，不能宣布压力为零。

全部活跃后台W13完整覆盖目标W13+W2，所有inactive后台都在目标结束后启动，前驱依赖成立。
实际后台W13至目标W13启动间隔整体119.16..724.60us；n1至少374.52us，n2至少333.61us，
n4第一场最短119.16us。因此“所有真实后台首个请求窗口已结束”尚未直接测得，
模型预测的request completion不能作为观测证据；本轮严格证明的是W13阶段覆盖和小幅净响应。
不据此给整个大M remainder设置新持续压力系数，也不外推到后台W2、阶段转换、其他前台M。
下一步需要用更长前驱/不同启动偏移覆盖后台更晚W13和W2，并冻结预测后采集；16T长历史W2
双峰问题仍待独立处理，完整计划与搜索验收未完成。默认planner未改变。


## 晚阶段混合后台已冻结并启动

本轮E类Lab扩展沿用原baseline、动态模型、entry/gather/publication参数和原生runner，不改生产。
prepare_late_background.py从原steady设计生成新目录late_background：前驱expert59/M529，
目标expert12/M8共用CPU312–319的8T team，最初启动的后台为85/14/115/42，即M1205/768/714/376，
同LLC CPU280–311四个8T team。n0/n1/n2/n4只表示初始启动数，不能当作目标时刻的活跃任务数。
原route计数不变，完整共享核祖先资源检查通过；inactive后台等目标，cohort外等cohort结束。

31个历史entry场景冻结预测均给目标W13/W2=125.44/59.75us，目标W13约10308.5..10314.0us开始。
各场景中n1后台85仍在W13；n2后台85在W13、14在W2；n4后台85在W13、14/115在W2，42已完成。
这只是模型预测的阶段分类，实际必须检查trace。新analyze.py逐pair记录W13/W2重叠、已完成
和转换/部分覆盖，不再要求所有后台W13完整覆盖；预测文件保持冻结，不按数据修改。
prepare与analyze均Ruff通过，原分析器仍负责数值/实际M/trace/isolation校验。

确认远端无旧benchmark、扩展SHA仍dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f，
且新目录不存在后，上传冻结目录并执行`bash tmp/joint_cost_model_20260911/late_background/run.sh`。
seeds611941/611942，两场串行，沿用H4096/F512 BF16/SVE256 Ntile16、NUMA3/80worker、
8T `(8,0,0,1,1)`、stage8/4MiB/owner1MiB/512KiB、4副本/216MiB scrub、固定workspace、
early_merge off、5warmup31pair。当前采集中，未读取新时间结果或重拟合；下一步以实际阶段覆盖评分。


## 晚阶段W13/W2混合对照完成：竞争增量小，完成时间仍有入口偏差

late_background/run.sh两场全部COMPLETE/exit0。analyze.py分别按冻结source/frontier/seed/build
核验31pair并评分；evaluate_cohort.py另评全部六个expert（包括inactive后启动任务）的阶段与完成时间，
Ruff通过。沿用上一节完整协议与seeds611941/611942，原始trace留Arm-codex-internal同目录，
本地analysis_session{1,2}.json和cohort_session{1,2}.json保留结果。

两场每个正式pair的阶段分类都与冻结预测一致：n1的85仍W13；n2的85/W13、14/W2；
n4的85/W13、14和115/W2、42已完成。目标整个W13+W2被对应阶段覆盖，无部分重叠样本。

| 最初后台数 | W13配对增量/us 两场 | W2配对增量/us 两场 |
|---|---:|---:|
| 1 | 0.78 / -0.38 | 0.20 / -0.12 |
| 2 | 0.08 / 0.13 | -0.71 / -0.58 |
| 4 | 1.07 / -0.03 | -0.93 / -0.59 |

n0 W13实测124.10/125.13us、W2 62.77/60.80us，固定预测125.44/59.75us。
本轮没有证据支持强持续竞争系数；但无预声明等价检验，不宣布真实资源压力为零。
与M48前驱的上一批相比，n0本身也变化，不能用竞争参数吸收前驱/会话基线差。

全部六expert×四条件×双GEMM阶段MAPE两场0.5018%/0.2480%。六expert cohort完成时间
（scheduled_compute原点至cohort最后W2，不含其他expert和final merge）四条件误差均为低估：
第一场169.91..211.86us（0.629..0.810%），第二场134.53..161.25us（0.444..0.693%）。
这些候选计划延后了不同的大expert，不能由其耗时差直接声称一般planner选择收益。

对目标完成时间按均值作可加分解（target_decomposition_session1.json），第一场四条件
总低估90.46/118.13/99.59/140.49us，其中前驱W13启动前的入口+gather合并区间贡献
64.09/87.89/76.80/94.87us；前驱W13自身贡献12.62..27.95us，handoff+目标gather贡献4.54..7.26us。
这里分解的是均值，与上面中位数完成误差不同，不把两个统计量相加。入口+gather尚未进一步
识别因果；下一步用已有worker trace分开worker到达与服务，再决定是否需要独立gather校准。
本批仅支持所测8T/大M/W13-W2稳定阶段，不覆盖阶段转换、16T和完整224expert计划。
默认planner与模型系数未修改，16T长期W2双峰和完整搜索验收仍未完成。


## 启动误差拆分：入口跨场漂移与大M gather基线/竞争同时存在

只用已完成late_background两场worker trace，未新增采集或修改参数。
`late_background/decompose_gather.py`按root任务逐worker做顺序替换：历史entry+预测service，
先换实测entry，再换实测service，最后换实测gather-to-W13间隙；均值分项严格相加为root W13
启动偏差，输出gather_decomposition.json，Ruff通过。顺序决定max-worker交互归属，
不是唯一物理因果分解；实测时间只用于诊断，不是自主预测输入。

M529前驱在n0/n1/n2/n4的实际减预测启动偏差中，entry项第一场38.79/55.66/34.30/41.47us，
第二场-1.86/-3.45/-1.46/1.37us；service项第一场24.86/31.62/41.73/52.48us，
第二场28.96/36.24/42.64/53.82us。setup项仅0.13..0.93us。入口误差跨场变化较大，
但服务项两场重复并随初始联合任务增加，不能只校准一个全局入口偏移。
M1205后台在n1/n2/n4的service项第一场62.74/96.69/180.78us，
第二场72.21/100.39/159.37us；同时存在大M基线误差和联合gather服务变化。

为先核查基线，再读取large8_baseline/train的两场隔离gather（expert85、8T CPU312–319，
原route嵌套M、同4副本/scrub/workspace/early_merge off协议），对8个worker各自31pair中位
服务时间取平均，与冻结read/write模型比较。结果isolated_gather_audit.json：
M384..1205平均worker服务整体低估约7.62..10.37%，例如M1205预测503.63us，实测545.17/549.47us；
M600预测250.66us，实测274.91/276.67us。这个统计量不是team envelope，不能直接加到完成时间。
小M12反而高估10.21..13.07%，因此不能全域乘同一系数。没有重拟合或从本检查宣称新模型通过。

下一步先补大M/实际gather分工下的无竞争服务项，冻结后在未用于gather拟合的已有形状做回顾检查，
再做新数据验证；固定该服务项后才拟合并发gather响应。初始entry跨场漂移仍独立保留，
不得用GEMM竞争或一个固定startup残差吸收全部偏差。16T长期W2与完整计划验收仍未完成。


## 大M gather单参数候选：回顾验证改善，边界点仍有小幅回退

新增独立gather_large_candidate/fit_check.py，原read/write系数冻结。按实际分工的
P>=8完整panel分配分支，对每worker的max(segments-1,0)拟合一个非负额外服务系数。
只用large8_baseline/train/session1的隔离expert85/8T/CPU312–319逐worker31pair中位数，
beta=3.887047182501948us；M1..84特征全零，M1..1205正服务检查通过。Ruff通过。
不改变entry/GEMM/竞争或生产接口；模型、完整结果和输入源码身份保存model.json/results.json/identity.json。

| 回顾数据 | 原worker MAE/us | 候选worker MAE/us | 原/候选条件finish MAE/us |
|---|---:|---:|---:|
| 原12留出第一场 | 21.081 | 6.845 | 22.292 / 7.218 |
| 原12留出第二场 | 19.688 | 6.146 | 21.863 / 7.731 |
| 额外6留出第一场 | 25.405 | 7.647 | 26.356 / 9.468 |
| 额外6留出第二场 | 24.872 | 7.290 | 25.009 / 9.300 |

conditioned finish使用实测worker到达，只检查服务项，不能称为自主计划预测。训练第二场worker
MAE19.969→5.917us。M104/M106四个shape/session的worker MAE均恶化约0.49us（约4.1..4.5→4.6..5.0us），
对应finish正偏约6.7..8.5us，保留此回退，不通过另加边界参数隐藏。其他留出逐worker MAE没有恶化。
这些数据是已采集的GEMM留出、此次未用于gather拟合，只能称回顾形状检查，不是新前瞻验证。
沿用原Arm NUMA3/80worker、H4096/F512 BF16/SVE256、4副本/216MiB scrub/固定workspace/
early_merge off、8T full stripes、5warmup31pair协议。下一步冻结该候选做新的gather形状检查，
再固定基线考虑并发gather；当前不接入planner。数学模型同步记录公式、限制及回退。


## gather候选的新形状前瞻验证已冻结并启动

prepare_gather_large_validation.py隔离同expert85/8T CPU312–319，真实route嵌套前缀：
M84/85/92/97/103/107/191/385/601/839/1001/1187。覆盖P>=8分工切换、原104/106回退附近，
以及中大M。共享核依赖资源检查通过。gather_large_validation/frozen_predictions.json在采前保存
旧/候选各8worker数值、模型与源身份及seeds611961/611962；beta与旧系数不再拟合。

采前门槛：两场分别要求M>=144组worker MAPE<=5%、实测entry条件下finish MAE<=12us，
且这两类MAE相对旧模型改善；每shape的worker MAE回退不超过2us；M<85预测完全相同。
这是允许保留隔离gather候选的门槛，不是自主planner/e2e验收。失败保留，不调整门槛。
准备与evaluate.py均Ruff通过；评估只读冻结数值和已校验compact，不导入/重新拟合候选。

确认远端无其他benchmark、新目录不存在、扩展SHA仍dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f后，
上传新目录并执行`bash tmp/joint_cost_model_20260911/gather_large_validation/validation/run.sh`。
两场13计划（12shape+anchor）串行；沿用NUMA3/80worker、H4096/F512 BF16/SVE256 Ntile16、
8T full stripes `(8,0,0,1,1)`、stage8/4MiB/owner1MiB/512KiB、4副本/216MiB scrub、
固定workspace、early_merge off、5warmup31pair。当前运行中，尚无新gather实测结论。
默认planner不变；并发gather、入口、16T尾项与完整计划/搜索验证仍待完成。


## 新gather第一场通过冻结门槛，第二场仍在运行

exec83522仍活跃，第一场已validated。取回compact后执行
`gather_large_validation/evaluate.py 1`，核对采前源/frontier/seed/扩展身份、31pair和实际worker。
M>=144组六新形状：旧worker MAE27.582us/MAPE8.446%，候选7.598us/3.032%；
实测entry条件finish MAE27.681→8.218us。大M两类误差门槛、相对改善、每形状回退<=2us、
M<85恒等五项全部通过。第二场尚未结束，不提前宣布两场验收或默认采用。

并行的本地旧数据诊断没有运行新benchmark：把冻结gather候选用于late_background两场已测
root任务，在实测worker entry下计算service剩余偏差，并记录实测peer gather worker重叠。
结果gather_large_candidate/joint_residual_diagnostic.json：M529 n0剩余5.43/9.53us，
n4为33.04/34.39us；M1205 n4仍134.14/112.72us。该诊断只定位并发服务缺口，
不把实测overlap接入自主预测、不拟合竞争系数；基线通过后仍需独立动态gather响应。
原协议、参数和默认planner不变；后续首先完成第二场冻结评分。


## gather无竞争候选两场前瞻通过；单系数并发响应仍有误差抵消

新12形状gather验证两场全部COMPLETE/exit0。evaluate.py 2按冻结源/数值/frontier/seed/扩展和
31pair评分：M>=144组旧worker MAE28.428us/MAPE8.811%，候选7.806us/3.055%；
实测entry条件finish MAE29.534→8.857us，全部五项门槛通过。与第一场共同支持保留
extra_panel_us=3.887047182501948的8T/H4096/M<=1205隔离gather服务候选；不代表并发或planner验收。
原始trace留Arm-codex-internal同gather_large_validation/validation目录，默认planner未切换。

新增独立gather_response_candidate/simulate.py，worker进度单位为无竞争service微秒，名义
需求r=(logical_read+logical_write KiB)/g0；同expert内部资源已包含于g0，不重复计压。
活跃worker速度v=1/(1+alpha*sum(other-expert active r))，按预测结束/预定启动事件更新资源。
不接收实测结束时间。无竞争还原、同expert不计额外压力、两等长peer解析解、错开无竞争、
无效系数等检查与Ruff通过。当前仅单LLC/8T root gather，不含GEMM压力、依赖激活或跨域耦合。

fit_replay.py只用late_background/session1/n1的代表性逐worker中位start/service目标，
固定上述无竞争候选，网格alpha=0..0.02步长0.00025，得到alpha=0.00175。
服务率的时间单位为us、logical需求为KiB；它是有效响应系数，不是测得带宽/硬件容量。
后续评分用每pair实测gather starts，只预测结束；属于已有数据的入口条件诊断，非自主/前瞻结果。
未拟合n2条件finish MAE两场35.525/38.528→12.080/12.410us，n4为59.245/59.176→28.259/21.434us。
但n4逐expert偏差显示抵消：M529高估23.65/23.01us、M1205低估73.23/49.95us，
M714高估20.35/15.93us。总体bias约-1..-2us不能支持模型通过；保留n4_expert_errors.json反例，
拒绝把单系数响应接入planner。下一步区分不同服务阶段/前台大小敏感度和共享带宽分配，
同时避免用实测overlap拟合后冒充自主需求模型。16T长历史双峰和完整计划验收继续保留为未完成。


## gather/W13跨阶段诊断：解释部分大任务低估，但直接叠加不成立

只读既有late_background两场trace。gemm_overlap_audit.json按root gather team envelope
与其他root W13求交；所有条件M1205均31/31有交叠，n1两场peer重叠总和约310/312us，
n4约1515/1395us。多个peer时长允许同时累计，不是墙钟长度或资源需求。
M529 n1/n2无peer W13交叠，n4约87/83us；M376无交叠。此结构与大任务残差有关，
但仅为观察，不证明W13请求导致全部减速。

新增独立simulate_gemm_diagnostic.py，保留原gather模拟器和alpha不变，允许外生pressure pulse。
用实测W13 start触发原模型的名义8MiB/161.18us首请求窗口，同一alpha=0.00175作用于gather，
不重新拟合。无竞争等原不变量与5us有限pulse解析解通过，Ruff通过。结果gemm_pulse_diagnostic.json。
这不是自主模型，也没有对W13作反向服务反馈，不能称为守恒资源分配。

n4 M1205 gather finish偏差从-73.23/-49.95us改善为-26.95/-3.53us，但M529高估增至
31.94/31.00us，M714高估增至41.98/37.78us；n4总体MAE28.26/21.43恶化到33.73/22.87us。
因此拒绝直接把GEMM pulse叠加到已用混合环境拟合的gather-only系数上。
原alpha可能已吸收部分跨阶段效应，且不同阶段的需求单位/响应尚未独立标定；本轮不唯一归因。
下一结构实验需分开gather-gather与gather-W13标定，并以预测gather完成触发W13、双方服务反馈
后检查自主时间线。不要用修好一个大M的oracle结果掩盖其他expert恶化。默认模型仍不改变。


## 分开标定gather-gather与gather-W13仍未解决大任务偏差

gather_response_candidate/fit_split.py只使用late_background/session1/n1，先验证每个pair的
M529 gather结束不晚于peer M1205 W13启动，再用M529的代表性worker向量拟合gather响应。
固定该系数后，用M1205标定名义W13 pulse响应；两者网格0..0.01步长0.0001。
无竞争模型保持已通过新数据验证的版本，tau=161.18us/Q8MiB固定。
得到alpha_gather=0.0016、beta_w13=0.0002。Ruff通过，split_results.json保留参数近优区间和逐expert误差。
这仍用实测gather/W13 starts，只是已有数据的分离诊断，不能称独立物理参数或自主验证。

未拟合n2条件finish MAE两场11.38/11.98us，n4为25.91/19.84us；
n4 M1205偏差仍-72.36/-49.22us，M529为+19.69/+18.96us，M714为+17.21/+12.77us。
相比原单系数，主要大任务低估基本未变，因此拒绝把两个系数视为问题已解决，不接入planner。
本轮支持转向更可控的背景形状/启动时间对照，检查需求随时间变化、前台敏感度和服务分配，
不能在相同n1/n4数据上不断加参再宣称泛化。没有新远端采集，原实验均已终止。


## gather后台形状×LLC位置发现网格已启动

E类独立Lab原因定位，prepare_gather_shapes.py复用cohort_bridge和依赖传递约简，
前台固定expert85/M1205、8T CPU312–319。三组各四个8T后台：small=[151/12,157/12,145/16,189/16]，
medium=[37/48,2/48,13/60,94/50]，large=[42/376,6/307,86/234,166/295]（expert/M）。
每组none/local/cross三个条件，local后台CPU280–311、cross CPU248–279；none后台等前台结束。
每组cohort外任务等五expert全部结束，前台与所有后台实际route计数不变，legacy nested expert13保持M60。
组间expert身份不同，不能把组间差异唯一归因于M；同组local/cross保持身份/输入/宽度一致。

静态共享核祖先资源检查和Ruff通过。gather_shapes/protocol.json采前记录分析口径：
按pair比较local-none/cross-none/local-cross的前台gather服务与envelope，保留每worker差异，
同时记录peer gather/W13/W2实际重叠。此网格为发现数据，不据此拟合后声称前瞻模型通过。

确认远端无其他benchmark、扩展SHA仍dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f、
新目录不存在后上传，执行`bash tmp/joint_cost_model_20260911/gather_shapes/run.sh`。
seeds611981/611982，两场10计划（9条件+anchor）串行，沿用NUMA3/80worker、H4096/F512
BF16/SVE256 Ntile16、8T `(8,0,0,1,1)` stage8/4MiB/owner1MiB/512KiB、4副本/216MiB scrub、
固定workspace、early_merge off、5warmup31pair。当前采集中；默认planner及所有模型参数不变。


## 后台形状实验采集中，配对分析器已准备

exec92321轮询仍活跃，第一场尚未产生完整compact；已取回stdout显示测量在推进，未重启或并行采集。
新增gather_shapes/analyze.py按协议检查frontier/seed/扩展、数值/isolation标记、31pair和前台CPU。
分别报告平均worker服务、最大worker服务、gather envelope、完成绝对时间、开始时间与arrival spread，
按local-none/cross-none/local-cross保留配对中位/MAD及全部差值。另记录各peer gather/W13/W2
与前台gather envelope的重叠总和，明确多peer可同时重叠，不当作物理需求或墙钟长度。
Ruff通过；尚未运行完整数据评分或作性能结论，等待原采集句柄正常结束。模型系数和默认planner不变。


后台形状第一场随后已validated，analyze.py 1按冻结协议完成评分，第二场仍运行中。
前台M1205平均worker gather服务配对增量（local-none/cross-none）为：small115.62/116.59us，
medium92.24/93.14us，large90.10/90.99us；local-cross分别+3.38/-0.60/-2.89us。
同LLC与跨LLC都出现明显服务增量，本轮主要影响不能直接按同LLC特有压力解释；但仅第一场，
也不能唯一识别DRAM原因。envelope/finish受entry差影响，已独立保留而不混入服务系数。
第二场与阶段重叠检查未闭合，不采用新参数或修改资源作用域。


## 后台形状第一场：减速分布于所有前台worker

继续轮询同exec92321，第二场仍未返回完整分析，不重启采集。第一场逐CPU配对服务诊断
保存workers_session1.json：small/local每worker中位增量约112..122us，medium/local约85..96us，
large/local约80..99us；cross相近。48个worker×条件组合中46个为31/31正增量，
另两个为30/31，说明不是少数单worker完成异常抬高team envelope。
该现象不同于16T长历史W2的少数慢worker双峰；仍不能仅靠此诊断唯一识别DRAM或其他共享资源。
第二场需要独立配对确认同/跨LLC结果，当前没有新模型拟合或采用。


## 后台形状×LLC两场完成：主要服务增量跨LLC仍存在

原exec92321正常exit0，两场均validated/COMPLETE。analyze.py 2核对31pair和运行身份后评分。
第二场前台M1205平均worker服务增量local-none/cross-none：small110.80/109.30us，
medium85.93/92.31us，large93.92/94.82us；local-cross为-1.32/-3.52/-1.11us。
两场共同支持：四个8T后台导致约86..117us服务增量，同/跨LLC主量级相近，不能将其仅编码为
同LLC压力。没有等价检验或硬件事件证据，故不宣布完全等价，也不唯一归因为DRAM。
small背景两场较重，同时其W13/W2均与前台gather重叠；不能按总M大小或简单任务数排序压力。

全部原始trace留Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/joint_cost_model_20260911/gather_shapes；
本地两场compact、analysis、第一场worker诊断保留。协议沿用前述NUMA3/80worker、真实route、
8T full stripes/4副本/216MiB scrub/固定workspace/early_merge off，seeds611981/611982。
下一模型结构必须允许跨LLC共享域的阶段需求影响gather，并分别核对每expert服务误差；
本发现不足以标定硬件容量，不能复用仅同LLC拟合的alpha冒充通用资源参数。
未启动新采集或改动默认planner，完整联合模型、16T和搜索验证仍未完成。


## 自主gather阶段衔接原型：去掉实测起点后旧系数仍无法迁移

新增独立unified_gather_candidate/simulate.py，输入root jobs、冻结逐worker g0、各stage T0与首请求窗口、
历史entry vector；预测最后gather worker结束+setup触发W13，预测W13结束+gap触发W2。
GEMM名义请求随预测事件开始/结束，gather速度由跨LLC的其他expert gather与GEMM需求决定。
当前GEMM自身仍用无竞争时间，尚无gather反向反馈或GEMM互争；只有root cohort，非完整DAG模型。
这是有意保留的未完成范围，不能声称已建立双向守恒资源模型。无竞争还原、同起点对称响应、
事件顺序和重复调用reset检查及Ruff通过，无生产改动。

replay.py只用31个历史entry场景，不输入新trace阶段起点或结束。系数沿用旧分离诊断的
alpha_gather=0.0016、beta_w13=0.0002；暂将beta应用到W13/W2名义请求，未拟合新数据。
先保存predictions.json，再评分已有gather_shapes数据，属于回顾检查，不是新前瞻验证。
目标M1205无后台平均worker服务中位偏差约-0.07..-5.95us，但有后台仍显著低估：
small约102..107us、medium约81..86us、large约60..63us；完成时间还叠加历史entry偏差。
因此旧系数在新背景形状下不具备可接受的迁移能力，不能仅凭自主事件链跑通接入planner。

下一模型工作需要在独立的跨LLC阶段需求证据上重新标定，并加入双向资源反馈，再验证完整DAG；
保留当前失败预测和各条件service/finish分离误差。当前没有新远端采集，默认planner与旧Lab模型均保留。


## 自主阶段响应单cell重标定及后台数量前瞻验证

unified_gather_recalibrated/fit.py保留隔离gather模型与alpha_gather=0.0016，仅用
`gather_shapes/session1/small local-minus-none`平均worker服务配对增量115.62us标定GEMM-to-gather
系数，31历史entry场景全部自主推进，beta网格0..0.01步长0.0001，得到0.0037。
两stage共用beta，名义Q/W13-W2窗口不变。既有GEMM固定T0/无反向反馈限制仍存在。
拟合增量115.05us；未拟合medium预测约74us、实测约86..93us，large预测约104us、实测约90..95us。
旧系数的大幅低估缩小，但仍有形状偏差；此处是发现数据回顾，不据此采用或宣称通用容量。
参数、扫描和结果保存在model_and_predictions.json/results.json，Ruff通过。

随后prepare_gather_count_holdout.py为同三组后台准备none/local1/local2/cross2，共12条件+anchor。
目标仍expert85/M1205，所有输入、expert组、8T位置和条带不变，只改变初始后台数量与同/跨LLC。
freeze_counts.py在采前用固定alpha/beta和旧entry场景保存mean-worker服务预测与身份。
相对none的预测增量：small n1/n2约38.47/66.75us，medium26.29/45.85us，large34.62/61.12us；
cross2接近local2。冻结预测不使用新实测starts，也不重新拟合验证点；验证仅针对目标gather响应，
不代表完整DAG/反向资源反馈或planner验收。静态资源依赖和Ruff通过。

远端无旧benchmark、扩展SHA仍dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f且
新目录不存在后上传，执行`bash tmp/joint_cost_model_20260911/gather_count_holdout/run.sh`，
seeds612001/612002两场串行，13计划、原5warmup31pair，NUMA3/80worker、H4096/F512 BF16/SVE256
Ntile16、8T full stripes、4副本/216MiB scrub/固定workspace/early_merge off协议不变。
当前运行中，尚未读取验证结果。默认planner未改变。


## 后台数量前瞻验证采集中，冻结评分器已准备

继续轮询原exec29092，第一场仍运行，未产生完整compact，不重启或并行采集。
新增gather_count_holdout/evaluate.py，只读取采前frozen_predictions.json数值；检查源身份、
frontier、seed、扩展、数值/isolation标记、31pair和前台CPU。分别报告预测/实测平均worker服务、
相对同组none的配对增量、绝对增量误差、配对MAD与原始差值，避免T0误差掩盖竞争误差。
Ruff通过；没有给未完成的数据打分或新增事后门槛，等待原句柄正常完成。
此验证只覆盖固定M1205/8T前台的新后台数量，完整双向/DAG/16T和搜索目标保持未完成。


## 后台数量留出第一场：从四后台拟合的响应高估一/二后台减速

exec29092第一场validated，第二场仍运行。第一次评分尝试早于rsync句柄35083完成，因compact
尚未落地报FileNotFoundError；等待该传输exit0后正常执行evaluate.py 1，未改变采集或评分参数。
九个有后台条件均高估配对增量，delta MAE23.006us；绝对平均worker服务MAE26.795us/MAPE4.691%。
small local1/local2/cross2预测38.47/66.75/66.68us，实测13.57/32.78/36.25us；
medium预测26.29/45.85/45.76us，实测10.47/30.27/35.73us；
large预测34.62/61.12/60.90us，实测7.55/33.49/39.26us。
不能用绝对服务较小百分比误差掩盖竞争增量不准，冻结候选尚不具备后台数量泛化支持。
第二场未闭合，不用第一场重拟合或追加threshold/knee。

并行本地检查既有gather_shapes两场peer GEMM，固定无竞争baseline的小M peer W13 MAPE
11.75%/11.53%，全部低估约10.7..14.1%；medium约2.54%/2.70%，large约0.80%/0.79%。
W2误差较小。产物unified_gather_candidate/peer_gemm_audit.json；这不是成对因果分解，
可能混合位置/历史基线与真实竞争，但明确表明当前固定GEMM时间的原型尚有阶段时间线缺口。
后续需处理双向反馈和受独立服务证据约束的非线性资源响应；本轮不采用新参数或切换planner。


## 后台数量两场完成：实测起点诊断不能修复减速高估

原exec29092正常exit0、两场validated/COMPLETE。第二场evaluate.py 2按冻结数值评分，
九个有后台条件仍全部高估，delta MAE20.519us，绝对服务MAE18.731us/MAPE3.232%。
small local1/local2/cross2实测15.59/37.15/38.76us；medium11.69/31.54/34.73us；
large11.30/40.04/40.97us，对照预测仍为上一节冻结数值。两场共同拒绝其后台数量泛化声明。
原始trace留Arm-codex-internal同gather_count_holdout目录，所有采集已经结束。

oracle_timing.py新增只用于诊断的外生回放：保持alpha/beta和名义burst长度/Q不变，
改用实测gather/W13/W2起点。两场18条件的预测减速相对自主版本最大变化小于0.4us，
仍保留原来的10..34us高估。因此在当前固定窗口假设内，起点错位不是主要误差解释。
但该外生干预有实测compute start早于预测gather finish的情况，逐expert-call计数已保留，
它不是可执行的自主模型，也不能排除真实请求窗口形状/持续时间变化。
Ruff通过，oracle_timing_results.json保留全部结果，冻结前瞻评分没有被覆盖或重拟合。

下一步应优先用独立资源服务证据约束压力到减速的非线性响应；单纯修正GEMM开始时间
不足以挽救本候选，仍需在完整模型中加入双向反馈。不能从本失败holdout追加经验knee后
再把同批数据称前瞻验证。默认planner、基线与原模型保持，16T与完整搜索目标仍未完成。


## 独立阶段服务PMU扩展已准备，尚未构建或采集

检查现有bench_eight_stage_demand与eight_stage_demand_native确认：旧工具只测单个8T独立
W13/W2、无gather接口，100ms持续窗口、NUMA3 16DDRC flux_rd/flux_wr、32B/event。
旧结果不能直接注入真实四副本/scrub/gather继承状态的短时模型。

新增E类独立目录independent_stage_service，将原未改JIT采集器扩为1/2/4个8T team。
每team独立B和输出地址，worker取CPU280..319最后8*teams个；controller240；共享常量A与route。
每次stage之间用跨team barrier对齐，记录calls为每team轮数，总stage吞吐必须乘teams。
M12/48×W13/W2×1/32副本×1/2/4team，加idle，共25条件。新增teams输入/输出身份，
数值与W2未触及route哨兵检查扩到每team，CPU检查扩到全部worker。普通vector分配约1.5GiB权重，
非HugeTLB。该实验检验持续服务曲线，不直接重现真实计划或证明gather/DRAM唯一因果。

stage_service.cpp、measure.py、build_smoke.sh和协议已准备，Python Ruff通过，
原生diff静态检查覆盖team地址偏移与CPU/线程数。依赖Python脚本快照已复制到独立目录。
尚未上传、编译或运行；下一步必须先原生数值/CPU/无PMU烟测，再PMU烟测与扰动控制，
不能先宣称正确或带宽结果。生产kernel、全局构建和默认planner未改。


## 独立阶段服务采集器构建与50条件烟测通过，正式控制/PMU测量启动

远端JIT源码SHA核对为1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629。
独立build_smoke.sh完成O3/C++17/pthread/SVE256构建，未调用全局build或修改生产扩展。
新二进制SHA ed2644e320069cdcceaa10e5d484d4b4dc04d5354bc35887f5f2aa1663c20154。
无PMU/有PMU各25条件完整一轮烟测，逐team常量数值、W2未写route哨兵与全部worker CPU检查通过；
analyze.load复核完整网格、计数enabled分母及running比例。编译stderr为空，身份和烟测结果本地保留。
这不是一般数值正确性证明或真实计划性能结论。

新增analyze.py保留每event自身enabled计数口径、idle扣除，总吞吐=teams*calls_per_team/window，
每team-call流量按总吞吐归一，不把同步轮次误当全部expert调用数。Ruff通过。
每次记录时间包含跨team barrier，不能直接拿其倒数代替吞吐，也不当作无竞争T0。

随后执行`bash tmp/joint_cost_model_20260911/independent_stage_service/run.sh`：
先无PMU控制seed612021，再两场PMU seeds612021/612022，25条件×(5warmup+31正式)，100ms窗口，
严格串行。目标CPU280..319末8*teams、controller240，NUMA3，M12/48×W13/W2×1/32副本×1/2/4team，
普通vector/非HugeTLB，共享常量A、独立team B/output和同步stage边界。当前运行中，
尚未输出正式服务曲线或采用任何容量参数。原始数据留Arm-codex-internal同独立目录。


## 独立PMU正式测量进行中，计数分析的关键不变量已检查

继续轮询exec87789仍活跃，未重启采集；最近取回control.stdout显示无PMU控制推进至21/31正式轮，
尚未取到全部三场终止数据。新增independent_stage_service/check_analysis.py在本地烟测产物上
验证完整25条件、多team总吞吐乘数，并以合成counter检查每event自身enabled分母、不同事件窗口、
running/enabled不足时拒绝；另拒绝重复cell、无效team数量和未终止日志。检查和Ruff均通过。
这些检查保证计数分析口径，不替代正式重复性/扰动证据，也不产生新的带宽或容量结论。
原始正式会话仍在远端独立目录顺序执行，默认planner和所有模型参数不变。


## 无PMU控制完成：同步开销限制吞吐，不能直接把曲线当内存容量

exec87789仍活跃，control.jsonl已complete，analyze.load校验25条件各31正式轮、数值/CPU与完整网格。
本地control_analysis.json记录每team与总stage吞吐，尚未使用PMU数据。
总吞吐相对1team扩展（2/4team）：W13/M12重复1份B为1.810/3.062，32份B为1.806/3.077；
W2/M12重复1份为1.681/2.585，32份为1.680/2.524。M48更接近线性，W13约1.95/3.69
与1.92/3.61；W2约1.90/3.46与1.88/3.34。

一份/32份B两种状态的扩展量级相近，说明仅凭吞吐曲线不能把短stage的次线性增长归为
DRAM饱和。跨team同步、调度及其他共享开销尚未拆开；还没有正式PMU与无PMU扰动对照。
记录区间包含一次完成barrier，循环还有其他barrier，calls/window反映完整同步协议吞吐。
本轮不会把这条受同步影响的曲线直接拟合成真实计划的硬件服务容量，PMU阶段继续原脚本串行运行。


第一场PMU随后complete，load核验全部25条件×31正式样本与二进制身份一致，第二场继续运行。
轮换32份B时读GB/s随1/2/4team为：W13/M12=47.69/80.58/142.24，W2/M12=41.20/61.63/98.43，
W13/M48=26.59/48.03/90.25，W2/M48=23.29/38.66/65.27。重复1份B时仅约0.07..0.72GB/s，
却同样存在明显次线性吞吐，进一步说明同步协议的影响不能忽略。
第一场PMU相对无PMU控制的记录区间中位差绝对值均小于0.45%；这是独立会话比较，不是同进程因果证明。
暂未观察到足以标定硬容量的稳定平台，不能取4team最高点当资源上限；首场预览保存pmu_session1_preview.json。
等待第二场完整计数与重复性，不调整模型参数。


## 独立阶段服务两场完成：流量可重复，但没有识别硬容量

exec87789正常exit0，control与session1/2均complete，各25条件×(5warmup+31正式)，共2700cell，
外加50烟测。analyze.py --root independent_stage_service --output analysis.json完整校验，
二进制身份一致、数值/CPU标记和counter有效时间通过。全部PMU对无PMU记录时间中位差绝对值<0.60%。
计数口径和多team吞吐单元检查仍通过；不是独立协议下的所有形式数值正确性证明。

32副本下读GB/s两场（1/2/4team）：W13/M12第一场47.69/80.58/142.24，第二场47.94/80.92/141.85；
W2/M12为41.20/61.63/98.43和41.99/59.97/98.22；W13/M48为26.59/48.03/90.25和26.40/47.89/90.77；
W2/M48为23.29/38.66/65.27和23.77/38.88/64.75。最高team数仍增长，未识别硬容量平台，
不把最高点当服务上限。重复1份B流量近idle而吞吐仍次线性，跨team barrier等协议开销不可忽略。

同round的32减1副本记录时间差保存reuse_contrasts.json。M12/W13随1/2/4team约2.7..3.7us，
M48/W13约6.5..7.2、15.9..16.5、21.8..22.2us；M48/W2约4.3..5.3、7.6..7.9、15.6..15.9us。
它是访问状态和同步协议下的配对差，不是纯DRAM stall，也不直接等于真实联合任务减速。

按控制器流量/总team调用吞吐估计，32副本M12/W13每team-call约7.45..8.04MiB，
M48/W13约15.44..16.37MiB；M48/W2约6.08..7.44MiB，且随team数变化。
这独立支持请求量不是所有M统一8/4MiB常数，但协议不同（持续/1或32副本/共享A/跨team barrier），
不能直接替换真实四副本/scrub短时模型的Q，也没有定位这些请求发生在哪个panel窗口。

所有原始JSONL、身份与stderr本地及Arm-codex-internal同独立目录保留，正式测量已结束。
结论是保留独立流量证据、拒绝从同步吞吐直接拟合硬容量；下一步需去除跨team同步影响，
并让真实阶段请求量/时间分布受独立证据约束。默认planner及旧模型参数未改变。


## 去掉跨team同步：独立推进版本烟测通过，正式测量启动

新增E类independent_stage_async，保持上一版M12/48、W13/W2、1/32副本、1/2/4x8T CPU位置、
共享常量A/route、每team独立B/output。每team现在有独立8worker barrier、iteration、completed与
预留records，TeamState按128B对齐，避免不同team状态共享cache line；不再等待其他team每次stage完成。
所有team完成64次lead-in后才ARMED。common观察窗口中分别统计team_calls，calls字段为均值兼容，
分析使用sum(team_calls)/window计算总吞吐，并核对均值字段与逐team整数计数一致。
记录interval为各team完成barrier在内的样本池，不包含跨team等待；不同team迭代数可以不同。

远端JIT SHA仍1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629。
独立build_smoke.sh正常exit0，二进制SHA c30ac66788bb7d50de5790bacd19caaa08ffc91407e87a65286ee577bfec7254。
无PMU/有PMU各25烟测条件通过全部team数值/W2哨兵/CPU及完整counter检查。
check_analysis.py的分母、计数复用拒绝、多team计数和不完整/重复/无效team输入检查通过；Ruff通过。
未修改生产kernel、全局build或旧同步版/旧数据。

随后执行`bash tmp/joint_cost_model_20260911/independent_stage_async/run.sh`，
控制seed612041、PMU612041/612042，25条件、5warmup31正式、100ms、NUMA3/80CPU范围，三场串行。
当前正式测量运行中，不从50烟测数值宣称吞吐改善或容量；旧同步版作为独立会话对照，
不是同进程配对实验。下一步先验证控制与PMU一致性及team间吞吐，再比较服务曲线。


## 独立team控制场完成：近线性吞吐支持原同步曲线受协议主导

exec93952仍活跃；无PMU control已complete。compare_sync.py复核完整25条件各31正式轮，
比较同shape/copies/teams的独立会话吞吐，同时保留逐team计数速率及spread。Ruff通过。
重复1份B时4team吞吐约4.00倍；轮换32份B时4team扩展：W13/M12=3.944，W2/M12=3.918，
W13/M48=3.825，W2/M48=3.914。对应旧跨team同步控制为3.077/2.524/3.610/3.342。
单team两版基本相同（约0..-0.17%），4team新版本总吞吐相对旧版本提高约5.94..55.28%，
这是采集同步协议变化，不是生产kernel或planner提速声明。

逐team完成次数的中位spread小，4team所有条件约0..1.13%，不呈现一组team明显饥饿后
由其他team抬高总吞吐的现象；逐team完整统计见comparison_control.json。
两种权重状态在去除跨team同步后均接近线性，强烈支持此前短阶段次线性曲线包含较大
同步协议开销，不能用它校准内存容量。PMU两场仍在运行，尚未有独立推进版正式流量结论。
不重启或并行其他benchmark，默认planner和模型系数不变。


## 独立推进第一场PMU完成：总流量继续增长，旧同步最高点不是容量

exec93952返回session1 complete，第二场仍运行。compare_sync.py session1与load验证完成，
完整原始计数、逐team counts及二进制身份一致；第一场PMU对本版本无PMU记录中位差绝对值<0.68%。
32副本下独立team读GB/s随1/2/4team：W13/M12=48.05/88.78/180.02，W2/M12=42.23/75.56/150.64，
W13/M48=27.33/52.73/96.24，W2/M48=24.26/44.10/76.63。4team吞吐扩展分别3.951/3.944/3.818/3.924。
一份B读流量仍约0.04..1.08GB/s，吞吐约线性。逐team计数没有明显饥饿，详情comparison_session1.json。

与旧同步版相比，W13/M12四team流量从142增至180GB/s，W2/M12从98增至151GB/s，
说明旧采集器的最高测得值不能作为共享容量。新范围仍无稳定平台，不据此标定硬上限。
新流量变化还受到异步相位和访问驻留影响，不能仅以总GB/s比例修正真实短时竞争。
pmu_session1_preview.json保留首场预览，等待第二场/控制完整对照，不修改模型系数或默认planner。


## 独立推进PMU两场完成，转向命令占用代理

exec93952正常exit0，独立版本control/session1/session2全部完整。analyze.py与compare_sync.py session2
复核所有team计数/身份/PMU有效时间。第二场32副本读GB/s的1/2/4team为W13/M12
47.88/88.56/180.76，W2/M12=41.14/75.82/150.23，W13/M48=26.92/52.91/96.85，
W2/M48=24.63/43.97/77.32；与第一场接近。4team吞吐扩展两场约3.82..4.00，
PMU相对无PMU记录时间差绝对值最大0.670%。每调用读MiB范围W13/M12=7.44..8.09、
W13/M48=15.51..16.93、W2/M12=3.48..3.90、W2/M48=6.15..7.76。
仍未识别硬容量平台，不把四team最大流量直接设成cost model上限。

现有linux_perf_event已有DDRC read_cmd=0x41、read_cmd_occupancy=0x80。measure_queue.py
在同一已验证native二进制上加入这两个事件，与flux_rd/flux_wr共64事件（16DDRC），
网格仅保留32副本M12/48×W13/W2×1/2/4team及idle，共13条件。
queue_smoke.jsonl一轮13cell完成，数值/CPU标记通过、64事件running/enabled全部1.0，
binary仍c30ac66788bb7d50de5790bacd19caaa08ffc91407e87a65286ee577bfec7254，Ruff通过。

随后执行`bash tmp/joint_cost_model_20260911/independent_stage_async/run_queue.sh`，
新无PMU控制seed612081、两场PMU612081/612082，13条件×(5warmup+31正式)、100ms，严格串行。
比较每event有效时间归一的command/occupancy速率和其比值，只称排队代理，不直接当CPU加载延迟，
不在没有时钟/事件语义迁移证据时换算ns。前述NUMA3/8T几何、独立team同步、共享A/独立B与输出保持。
当前运行中，原始数据保留同目录独立queue前缀；默认planner与参数未改。


## 命令占用两场完成：吞吐近线性不代表排队代理不增长

exec82048正常exit0，queue_control与queue_session1/2全部complete，各13条件×(5warmup+31正式)，
共1404cell加13烟测。analyze_queue.py逐cell检查64事件覆盖、每event有效时间、完整round/grid、
数值/CPU与逐team整数计数；二进制身份一致。合成计数检查加权比值与不同event分母通过，Ruff通过。
指标先按每event enabled归一，再sum(occupancy_rate)/sum(read_cmd_rate)，保留逐controller比值；
不对idle比值做减法，不换算CPU加载ns，原始样本全部保留。

| stage/M | 1team占用/命令 两场 | 2team 两场 | 4team 两场 |
|---|---:|---:|---:|
| W13/M12 | 35.716 / 35.637 | 45.002 / 45.027 | 80.221 / 79.397 |
| W2/M12 | 29.275 / 30.053 | 34.763 / 35.801 | 56.184 / 58.853 |
| W13/M48 | 32.696 / 32.998 | 39.088 / 37.150 | 60.730 / 59.347 |
| W2/M48 | 26.835 / 27.746 | 29.239 / 30.542 | 37.431 / 37.835 |

64事件PMU对新无PMU控制记录时间差绝对值最大约1.02%，第一场W13/M48两高并发点约0.92/1.01%，
第二场均小于0.37%；这是独立会话对照，不自动归因为计数器开销。不能隐藏此大于前32事件的差异。

在同一独立推进协议下，吞吐仍接近线性，但占用/命令代理随并发显著增加且并非简单线性增量。
这是支持延迟敏感前台可能在未出现硬吞吐平台时减速的独立证据，不是对真实gather因果的完整证明。
不同stage/M的代理-带宽对应关系也不同，不把总GB/s作为唯一队列预测输入。
下一联合模型需要分开吞吐服务和排队延迟响应，并用真实短时协议检查映射；当前数据不能直接作为
预测计划未来PMU值的oracle输入或校准CPU延迟。固定Q/线性减速候选的失败仍保留，未部署新模型。
所有采集已结束，数据在Arm-codex-internal同independent_stage_async目录及本地queue前缀文件。


## 独立占用代理的二次count插值及新3team检查

queue_curve_candidate使用每个(stage,M)的session1 n1/n4两点固定q(n)=a+b*n^2，
b=(q4-q1)/15，a=q1-b。仅限相同shape、8T、32副本、稳态独立team的n1..4；
不是物理定律、异构压力或瞬时gather延迟。n2不参与拟合，但此前已查看，因此只称回顾检查。
四组n2的二次插值MAE第一/二场0.390/1.073代理单位，线性count插值为3.278/3.259。
模型和输入身份保存在model.json/identity.json，checks.json保留全部点；n1/n4插值精确且b>0。

新n3预测在采前冻结：W13/M12=59.4524、W2/M12=43.6263、W13/M48=47.6475、W2/M48=32.4862。
queue_count3复制独立team采集器，原生仅扩展teams输入校验允许3，数组容量/计算/输出/计数不变；
新grid为n1/n3/n4控制与验证、两shape两stage共12cell+idle，32副本。预测不按新数据重拟合。
正式验证仍需无PMU控制与两场PMU，不能只用烟测数值判定曲线通过。

前批已终止，新目录确认不存在后上传。build_smoke.sh已启动独立构建和无PMU/64事件PMU烟测，
原生数值/CPU/逐team计数仍是先决条件；当前构建烟测中，正式run.sh尚未启动。
计划正式seeds612101/612102、无PMU612101，原NUMA3/100ms/5warmup31pair协议保持。
默认planner未改，完整联合模型与真实瞬时需求映射仍未完成。


queue_count3构建烟测随后exit0；新binary SHA5a18117795979402d0136bee1ee76a2130499047ad4252308e58a8de9a2c99eb。
无PMU/PMU各13cell烟测经analyze_queue.load确认包含3team、数值/CPU/逐team计数和64事件有效覆盖。
正式`bash tmp/joint_cost_model_20260911/queue_count3/run.sh`现已启动，无PMU控制再两场PMU严格串行。
尚未读取新count3占用结果或重拟合预测，默认planner保持。


## 新3team点误差小，但同期4team控制点显示占用曲线不完全稳定

exec75412正常exit0，queue_control/session1/session2均complete。evaluate.py逐项核对冻结model hash、
seed/build、64事件与完整grid，直接用原a/b，不用新n1/n4重新标定。
四个n3点预测/实测第一/二场：W13/M12=59.452/59.202/59.418；W2/M12=43.626/44.571/42.258；
W13/M48=47.647/49.603/49.432；W2/M48=32.486/33.210/31.554。
n3 MAE0.968/1.030代理单位，MAPE2.166%/2.465%，最大相对误差3.941%/3.611%。

不能据此宣布整条曲线稳定：同期W13/M48 n4第一场实测51.862，相对冻结60.730低约14.6%，
预测误差+17.10%；第二场回到61.876。W2/M48 n4第二场预测误差+7.44%。
全部控制点与MAD保留，未用较好的n3结果掩盖控制失败，也未按控制点重新平移曲线。
本轮只支持新n3插值的局部表现，独立稳态占用模型仍有会话/相位/布局等待辨识因素，
不能作为已可靠的异构/短时排队预测器或直接换算gather延迟。

同时只读定位实际gather实现：生产固定team路径调用common文件的
`gather_pack_a_reorder_sve_hybrid`，使用m12/m8分工与K条带；sve_bf16/packing.cpp的同名简单
接口不是该完整混合分工的直接替代。既有test导出每次分配并清零输出，不能直接拿其PMU计数
代表纯gather服务。下一步若独立测gather，必须保持实际分工、路由和预分配输出，并验证与生产
参考一致，不能直接替换为普通memcpy或忽略分配成本混入。当前未新增gather实现或采集。
全部远端测量已结束，默认planner未变，完整目标仍未完成。


## 直接gather探针前置：生产混合分工快照和正确性检查

E类独立gather_native_probe，从当前common/fused_moe_bf16_tiled.cpp按符号边界原样提取
GatherPackAPlan/K条带分工、m8/m12 gather和hybrid入口到gather_snapshot.h；源与片段SHA见identity.json。
不修改生产代码，也不使用只支持m8的packing.cpp接口代替实际hybrid路径。

check.cpp用散列16bit输入、散乱route/topk6、H37→Kpad40和H4096两种列形状，
M1..12、13/17/23/24/48/60/84/85/96/97/104/106/529/1205，width1/2/4/8/16，
共260case，检查逐元素packed布局和padding、前后输出guard，以及每个物理row/K位置的
worker所有权恰好一次；再并行执行helper并对照参考。该测试不含性能计时。
本地`c++ -std=c++17 -O2 -pthread .../check.cpp -o .../check_local`及执行通过260case。
Arm用g++ O3/pthread/armv8.2-a+bf16+sve/SVE256独立编译，在NUMA3/CPU240–319范围执行同组检查。
同一份源的独立编译不保证与生产translation-unit的机器码/内联或性能相同，未来必须做基线迁移检查。

下一步构建预分配输入/输出、计数窗口外准备/验证的gather服务探针；现在还没有gather PMU结果，
不能用带分配/清零的test导出时间替代纯gather。默认planner、内核及全局构建保持不变。

Arm检查句柄随后正常exit0，check_arm.stdout确认同260case全部通过。当前没有运行中的远端采集。


## 预分配直接gather采集器两种编译烟测通过，时间迁移尚未成立

gather_native_probe/gather_service.cpp以已验证原样hybrid快照构建gather-only循环，8worker
CPU312–319、controller240；输入/输出32槽一次分配并预触页，初始化、线程创建、64次lead-in、
每副本逐元素验证均在100ms计数窗口之外。每次调用只有team内部barrier，单独记录worker区间，
不把完成barrier作为worker服务时间。实际expert85/layer4的1205条route IDs已从原route提取，
M12/48/529/1205使用该前缀，topk6；输入为确定性16bit模式，不是原hidden数值。
每条件比较1/32输入与输出槽，1槽不等于真实冷状态，32槽也不是四副本scrub协议。

v1 build_smoke.sh无PMU/PMU各9cell完成，全部副本packed/padding/输出guard、CPU和64事件检查通过，
binary SHA75ad781d63d570ef983f3b5f8ba680231977f86b160b05690afcae29ba5af0c1。
烟测平均worker时间：M1205重复1槽约329us、32槽约502..504us；M12约2.9..3.0us，
与真实路径隔离gather约M1205 545..549us、M12 4.4..4.6us不同，不能直接替换真实T0。

为检查编译特化影响，保留v1不变，新增gather_impl.cpp与gather_api.h，将驱动与gather
分开translation unit编译且不使用LTO；v2同时拒绝不支持的stride7（真实路由固定，所有实验stride1）。
build_smoke_v2.sh正常exit0，binary SHA0045fe1b13423770ba949c7df23f00d941c909596241aef94c9a3f86aa5a26df。
无PMU/PMU各9cell同样全部通过，v2_smoke_summary.json保留检查值。M1205单槽约328..329us、
32槽约504us，M12仍2.9..3.05us；分开编译没有消除烟测所见时间差，因此不能单独归因于
call-site常量特化。正式重复性和协议匹配尚未完成，这些时间只是烟测，不作性能改善声明。

当前未启动正式gather流量采集。下一步需要区分缓存/预触页/首调用与持续循环、完整worker池
及测量口径影响，或将此工具严格限制为独立访存对照；保留真实trace拟合的gather基线。
生产kernel、全局build和默认planner未变，所有本轮进程已结束。


## 同worker热/冷对照与实际编译配置核查

新增temperature.cpp，持久8worker固定CPU312–319，controller240；固定同一16MiB输入与预分配
packed输出、实际expert85路由前缀。每个cell先同一helper执行64次复用，再比较直接测量与
8worker各自读取27MiB、合计216MiB scrub后测量；scrub、barrier和逐元素验证在worker计时区间外。
M12/48/529/1205×warm/scrub按round随机，guard和每worker CPU检查保留；该缓存对照仍不等于
真实80worker调度、MoE scrub代码或完整调用协议。

首次O3/SVE烟测8cell全部通过，scrub使M12约2.85→4.70us，M1205约328.09→402.95us。
核查setup.py并只读远端build/temp.linux-aarch64-cpython-312/build.ninja发现重要配置差异：
MoE common的最终post_cflags为`-fopenmp -O2 -std=c++17 -march=armv8.6-a+bf16+i8mm`；
前面的Python默认O3被后置O2覆盖，common未启用SVE自动向量化。此前独立采集器使用O3/+sve，
不能视作匹配生产编译选项。没有重建生产扩展。

另以O2、fPIC、fopenmp、fno-strict-overflow、armv8.6-a+bf16+i8mm编译temperature_matched，
同8cell烟测数值/CPU通过：M12 warm/scrub约3.17/4.91us，M1205约345.54/425.51us。
相较O3变化存在，但仍未消除真实M1205约545..549us的差距，不能单独归因于编译或缓存。
随后启动matched版本两场31正式+5warmup随机对照（seeds612141/612142），当前运行中，
不从单轮烟测拟合参数；真实trace gather基线保留。还核查record_phase在分配/写trace前捕获end，
gather计时内未发现大规模route-out清零，避免凭猜测把额外清零加入模型。


matched温度对照两场均exit0、每场8cell×(5warmup+31正式)=288cell，通过全部packed/guard/CPU与
完整round检查。temperature_analysis.json按每次8worker服务均值，再对31pair取中位：
M529 warm142.800/142.824us，scrub210.206/212.379us，配对增量67.376/69.838us（MAD1.850/2.530）；
M1205 warm340.661/341.990us，scrub412.515/415.961us，增量72.375/73.395us（MAD4.936/4.096）。
M12 warm3.171/3.171us，scrub3.681/3.860us；M48 warm12.689/12.700us，scrub13.580/18.227us，
小M尤其M48的scrub效果跨场不同。真实MoE使用不同scrub指令/worker池/生命周期，不能认为
本读scrub等价于真实冷状态，也不能把这些独立时间替换真实gather基线。

证据支持编译选项和访问状态都需要匹配，但并未闭合剩余时间迁移差异。
仍保留真实trace拟合的gather成本；未来独立资源实验应使用common匹配的O2/+bf16+i8mm
实现单独编译，而非O3/SVE自动向量化版本。当前所有本轮进程已结束，没有生产构建或planner变更。


## common匹配编译的持续gather：烟测通过，独立流量测量启动

build_smoke_matched.sh把gather_impl.cpp单独编译为O2/fPIC/fopenmp/fno-strict-overflow/
armv8.6-a+bf16+i8mm对象，不启用SVE自动向量化；驱动独立O2/SVE编译只为CPU向量长度检查，
不包含gather定义、无LTO。binary SHA79e1c1f79b09fe0177dd83a061fc94c44027801cdde51616211cf8489aca2a67。
无PMU和PMU各9cell烟测complete，全部副本数值/输出guard/CPU/64事件/完整grid检查通过。
M1205持续32槽平均worker约526.4..526.7us，比旧O3/SVE约504us接近真实545..549us，但仍不等价；
M529约233..234us，M12约3.2..3.3us。只作为烟测观察，不作性能结论或替换T0。

新增analyze.py分别记录worker服务、控制器read/write、吞吐归一的每调用流量和占用/命令代理。
每event使用自身enabled时间；近idle时允许command代理缺失，保留原始而不拟合噪声比值。
Ruff通过。run_matched.sh现已启动无PMU控制seed612161，再两场PMU612161/612162；
9cell×(5warmup+31正式)、100ms，NUMA3、controller240/8worker312–319，全部串行。
M12/48/529/1205的expert85实际路由前缀、固定H4096、1/32预分配输入及输出槽、合成16bit内容，
准备/清零/验证均在计数窗口之外。当前运行中，尚无正式流量结果。
这是独立稳态gather对照，不是四权重副本/真实MoE scrub/80worker池的T0校准；保留真实基线。


## 直接gather第一场PMU完成：大M输入/输出复用状态显著改变读写流量

exec89397第一场complete，第二场仍运行。analyze.py 1按64事件自身enabled时间、完整grid、
数值/CPU与binary身份核验；另对零命令保留None和不同event分母加权作合成检查通过。
第一场PMU对无PMU平均worker服务差绝对值最大约0.37%。

M12/48/529/1205的1槽服务约3.20/12.73/144.50/343.78us，32槽约3.30/14.25/227.10/525.33us。
idle扣除后1槽read/write GB/s分别约0.158/0.007、0.119/0.005、0.474/0.032、4.944/0.247；
32槽分别0.850/0.012、4.482/1.527、30.522/14.655、34.836/17.178。
输入与输出槽一起轮换，不能把差异单独归因于输入cache miss；尤其大M有显著写流量，
不能只使用一个统一只读压力常数。近idle小M代理可能主要是系统活动，不能据其比值拟合延迟。
第一场结果保存matched_analysis_session1.json，尚未完成重复性；不替换真实T0或声称真实短时Q已确定。


## 匹配编译的直接gather两场完成：读写流量重复，输入/输出状态尚需分离

exec89397正常exit0，matched_control和两PMU场全部complete，各9cell×(5warmup+31正式)=324cell，
共972cell。analyze.py 2复核64事件、逐team计数、数值/CPU/完整grid及同binary；两场最大PMU/control
平均worker时间差约0.37%。第二场1槽M12/48/529/1205服务3.20/12.73/144.47/343.99us，
32槽3.30/14.28/227.08/525.53us，与第一场接近。

32槽大M的read/write GB/s第一/二场：M529=30.522/14.655与30.499/14.665；
M1205=34.836/17.178与34.756/17.121。按控制器速率/完整调用吞吐估计每调用read/write MiB：
M529约8.002/3.842与7.995/3.844；M1205约19.215/9.475与19.282/9.499。
这些读流量约为有效输入大小的两倍，写流量接近packed输出大小，与写分配/RFO贡献相容，
但并未按地址归因，不能把这个解释当唯一证明。当前同时轮换input/output，尚未分离两者影响。
小M32槽仍大量复用缓存，不能把“32槽”统一称作全冷。

下一步将输入与输出复用分别控制，检验read/write及服务响应，避免直接给所有gather套一个
固定字节放大系数。保留真实trace的T0，本独立持续协议只提供需求/服务证据，不直接代表短时计划。
全部本轮采集已结束，原始结果本地及Arm-codex-internal同gather_native_probe目录保留。
默认planner和当前模型参数未改变，完整联合与搜索目标继续保持未完成。


## 输入/输出复用分离对照已启动

新增E类gather_io_states：固定M529/M1205、实际expert85 route前缀、8T CPU312–319，
input slots与output slots各取1/32，四种组合独立控制。保持同一预分配地址空间和匹配common的
O2/+bf16+i8mm gather对象；驱动新增output_copies参数。最后写入每个输出槽的input slot由
worker0在完成barrier后记录，验证按真实最后输入核对所有输出槽与guard，避免异步槽周期误判。
原gather_native_probe源、二进制和数据不变。

build_smoke.sh正常exit0，binary SHA f47b8336663f1980ba360994e378c7b7b2c88a83df3b13d75a2950eec62dd577。
无PMU/PMU各9cell（8活动+idle）complete，load复核四种组合、数值/CPU/完整grid/64事件有效覆盖，
Python Ruff通过。烟测不用于带宽或性能结论。

随后执行`bash tmp/joint_cost_model_20260911/gather_io_states/run.sh`：无PMU seed612181，
PMU612181/612182，两场加控制严格串行；9cell×(5warmup+31正式)、100ms观察、NUMA3，
准备/清零/64次lead-in和全部验证均在计数窗口之外。当前运行中。
分析将比较同输入状态下改变输出槽，以及同输出状态下改变输入槽的服务、read/write和占用代理，
检查交互而非预设统一流量倍数。此独立持续协议不替换真实trace的T0，不直接视作瞬时联合计划。
默认planner与模型参数未改变。


## IO状态控制与首场PMU：服务交互为正，流量不能按固定放大倍数相加

exec9233控制和第一PMU场已complete，第二场仍运行。contrasts.py按同round计算四个主效应
和交互(32,32)-(32,1)-(1,32)+(1,1)，保留配对中位/MAD与全部差值；Ruff通过。
控制场M529四状态(输入,输出)=1/1、1/32、32/1、32/32的平均worker服务分别144.660/187.031/
173.446/231.641us；M1205为345.916/441.026/436.377/543.205us。
服务交互中位M529=+15.050us（MAD2.499）、M1205=+11.715us（MAD4.207），不是简单可加。

首场PMU确认同样服务交互约+14.131/+11.007us。M1205四状态read/write GB/s分别
4.248/0.241、17.694/18.686、30.651/3.975、33.973/16.651。仅轮换输入时输出固定仍有写回，
仅轮换输出时也有额外读取，说明缓存驻留/写分配等可能相互影响；不作地址级唯一因果解释。
首场每调用read/write流量交互为负（M1205约-1.742/-1.026MiB），同时服务时间交互为正，
更不能把总逻辑字节固定乘一个系数直接当成本。每调用流量仍是速率/吞吐估计，非精确单次计数。

analyze.py 1复核64事件、完整grid和二进制身份，PMU相对控制服务差绝对值最大约0.67%。
等待第二场重复性，真实T0不变；本独立持续缓存状态不直接替代真实短时计划的状态。


## IO状态两场完成：正服务交互重复，拒绝固定可加字节成本

exec9233正常exit0，control和两场PMU均complete，各9cell×36轮，共972cell。
analyze.py 2与contrasts.py session2复核全部64事件/数值/CPU/grid/binary；第二场PMU/control
平均worker服务差绝对值最大约0.90%。

第二场M529四状态1/1、1/32、32/1、32/32服务143.71/185.93/173.06/231.81us；
M1205为345.09/439.98/432.47/542.56us。配对服务交互两PMU场M529=+14.131/+17.951us，
M1205=+11.007/+14.627us，与无PMU控制正交互方向一致。
第二场M1205四状态read/write GB/s=5.158/0.238、18.691/18.784、30.470/3.372、33.990/16.674，
每调用read/write交互仍为负约-1.657/-0.842MiB，第一场为-1.742/-1.026MiB。
因此服务非可加性不能只通过增加固定逻辑字节放大项解释；输入/输出复用与读写服务需同时考虑。
这不唯一识别缓存层或写分配机制，不能把控制器总计数按地址直接归属。

全部数据本地及Arm-codex-internal同gather_io_states目录保留，测量已结束。
本轮保留直接gather读写/状态证据，不替换真实trace的T0或新增经验interaction时间项。
下一步应在可控GEMM后台下测同一前台的需求与服务，检验读写/排队响应如何迁移到实际联合运行，
而不是继续只拟合无竞争状态。默认planner不变，完整联合与搜索验收仍未完成。


## 可控W13后台下的直接gather：协调/PMU烟测通过，正式测量启动

新增E类gather_with_gemm，复用已验证foreground二进制f47b8336...（真实route前缀、匹配common
编译的gather、32输入/输出槽）。后台为独立8T M12/W13进程，0/1/2/4team，CPU为312之前的
8*n个核（n4即280–311），与前台312–319不重叠。后台团队独立推进，B副本32为主，另加n4/B1
控制；B1在gather流量下不预先认定为cache热。两个进程controller均240、同NUMA3。

14cell=idle、M529/M1205前台各n0/1/2/4(B32)、M1205+n4/B1，以及n1/2/4 B32与n4 B1的
background-only控制。两个native均到ARMED且已经过64次lead-in持续运行后才启用64个DDRC事件；
GO顺序发送，各自100ms观测，DONE后继续工作直至全部计数器停用再ACK。各自native统计窗口的
绝对起点不完全相同，不声称完全相同时间戳；PMU窗口中双方持续工作。控制器总流量不按前台
调用数全部归给gather，也不直接减background-only流量忽略后台速度/驻留变化。

后台源码仅移动原独立stage采集器CPU范围，binary SHA a4d4a44f586d5f5ae466b608b65905679d193c268a11b0ceed05f62d87e9e667。
无PMU/PMU各14cell烟测complete，原生数值/CPU/身份与前后台presence检查通过，64事件running/enabled=1。
Python Ruff通过。foreground不改、生产扩展与全局build不改。

正式`bash tmp/joint_cost_model_20260911/gather_with_gemm/run.sh`已启动：无PMU控制612201，
PMU612201/612202，14cell×(5warmup+31正式)、100ms，全部串行。记录前台worker服务、后台每team
调用数、聚合读写和占用/命令，不在本发现网格上拟合后宣称前瞻验证。
这是独立进程的稳态原因定位，不是实际MoE的同进程调度/阶段状态，真实trace的T0保持不变。
当前测量中，尚无正式联合服务结果。


## 可控后台无PMU控制完成：前台非线性减速及后台反馈可分辨

exec76330仍活跃，control.jsonl已complete，14cell×31正式轮及5warmup完整。
analyze.py核对前后台角色、shape/copies/计数、数值/CPU和完整round，Ruff通过。
前台M529独立平均worker服务239.470us，1/2/4个B32后台配对增量10.424/34.039/104.345us，
MAD0.755/1.094/1.465us；后台总调用率相对同n的background-only下降约0.709/1.584/2.817%。
M1205独立544.229us，增量21.420/74.301/233.346us，MAD0.771/1.131/1.276us；
后台吞吐下降约1.234/2.111/3.351%。

M1205+n4/B1前台配对增量仅0.125us（MAD0.778），后台吞吐下降0.390%。
相同team数和GEMM形状，权重复用状态改变了前台减速，反对仅任务数或只按占用核数计压。
B1是否实际DRAM近零必须由PMU确认，不先把它叫作完全cache命中；背景吞吐反馈也不能忽略。
目前仅无PMU控制，正式PMU两场尚未收齐，不据此拟合或宣称真实短时计划映射通过。

分析器分别报告fg service、bg throughput和聚合PMU，不将总counter bytes除以前台calls后
冒充前台流量；背景服务变化意味着简单减background-only流量也不成立。
数据与control_analysis.json保留，默认planner及真实T0保持不变。


## 可控W13后台两场PMU完成：前台相对响应跨M接近，后台反馈与占用代理同步变化

exec76330正常exit0，control/session1/session2均complete，共14cell×36轮×3=1512cell，
另28烟测。analyze.py完成全grid、native roles/shapes/counts、数值/CPU和64事件有效时间校验。
前台服务相对无PMU控制的两PMU场最大差约1.2%以内，完整值在原始summary中保留。

M529 n0服务两场239.080/239.166us；B32 n1/n2/n4配对减速为
10.905/33.126/105.363us及10.794/34.455/107.015us。
M1205 n0为542.847/544.289us；减速22.524/73.683/238.858us及20.913/74.351/242.079us。
两种M的相对减速相近：n1约4%、n2约14%、n4约44%，但不从两M直接推广至小M/K条带分工。
后台总吞吐在大M前台下下降约1..3.6%，不能假定后台服务完全不受前台影响。

M1205的聚合占用/命令代理从n0约44.4升至n1约51.5、n2约62.6..62.9、n4约118.0..118.7；
M529从43.5升至50.7、61.7..62.1、116.1..116.6。n4聚合read约194..198GB/s、write约11..12GB/s。
B1的n4控制下，M1205减速仅+0.227/-0.317us，聚合read/write和占用代理接近前台独立值，
而后台仍持续计算；不能把“4个任务”作为不区分访存状态的压力。
该实验支持非线性排队/供给响应与较弱后台吞吐反馈同时存在，不等于已经证明唯一微架构因果。

全部原始数据与control/session1/session2_analysis.json保留在本地及Arm-codex-internal同目录。
下一步可以只用一个前台M和部分count标定资源代理与敏感度，保持真实T0，验证未拟合M/count及
真实瞬时阶段；目前未拟合新参数，也未把这些稳态双进程数据当作真实MoE调度预测器。
所有本轮采集已结束，默认planner不变。


## 分层gather队列候选：单M/部分count标定，回顾跨M误差小

新增gather_queue_model，固定已通过独立新形状验证的真实gather T0。
只用gather_with_gemm/session1/M529的n0/n1/n4 controller代理拟合
q(n)=q0+a*n+b*n²：q0=43.47618339932979，a=3.4933404579739666，b=3.6666581443240225。
再只用同场M529/n4配对减速标定k=0.006056281924737621，
T(M,n)=T0(M)*(1+k*(q(n)-q0))。q是该稳态混合读写环境的有效代理，不是CPU ns、硬容量或一般队列定律。
仅限8T、M529..1205、32input/output槽、同质M12/W13 B32 n0..4；不把n直接用于异构或热B计划。

未参与拟合的M1205 n1/n2/n4两场配对减速误差为+1.27/-1.71/+2.58us和+2.89/-2.38/-0.64us；
M529/n2误差-1.72/-3.05us。此前已查看数据，因此这是回顾迁移检查，不能称前瞻。
model.json/checks.json/next_predictions.json保留参数和逐点结果。predict.py检查n0精确还原、
单调响应、15个冻结预测一致，拒绝域外M/count、NaN/inf/负参数，Ruff通过。

新gather_queue_holdout已准备：M768的n0/1/2/3/4与B1/n4控制，M529/M1205的新n3和各自n0控制，
配套background-only n1/2/3/4 B32、n4 B1及idle，共16cell。M768/n3冻结预测service439.701us、
增量91.651us、代理86.956；所有预测在采前保存，不按新控制点更新T0或q0。
原生仅扩展foreground允许M768、background允许n3；gather/JIT执行及预分配布局保持。
新目录独立构建，无PMU16cell烟测complete且数值/CPU通过，当前64事件烟测运行中；正式run.sh尚未启动。
计划seeds612221/612222，真实默认planner不变。后续仍须瞬时/异构/双向时间线和完整搜索验证。

gather_queue_holdout的64事件16cell烟测随后complete，数值/CPU/运行比例校验通过。正式run.sh已启动，
控制612221再PMU612221/612222，16cell×(5warmup+31正式)，100ms，三场严格串行。尚无前瞻结果。


## M768/n3前瞻控制场完成：部分新点好，但存在基线/竞争抵消

exec10699仍活跃，control.jsonl已complete。evaluate.py核对冻结model hash、seed及load的全grid/
native检查，按原预测分别评分service、paired delta和后续PMU proxy，Ruff通过；不以新n0更新模型。
控制场新竞争点service MAE10.727us/MAPE2.206%，delta MAE9.063us，最大delta误差21.309us。
M529/n3 delta误差+3.664us（MAD2.318）；M768 n1/n2/n3/n4为-7.206/-8.417/+1.132/-12.647us；
M1205/n3为+21.309us（MAD5.034）。

独立T0预测也漂移：M529/M768/M1205 n0的service误差+3.312/+12.516/+14.743us。
M768/n4总service误差仅-0.633us，实际是T0高估与delta低估抵消，不能据总时间小误差认为该层通过。
M768/n3 delta较准但其service仍高12.686us；M1205/n3总service高35.225us。
保留所有分项，等待两PMU场检查资源proxy、敏感度和重复性，不据控制数据重新拟合。
默认planner与真实基线不变，完整目标仍未完成。


首场PMU随后complete，evaluate.py session1按冻结值评分：service MAE8.971us/MAPE1.803%，
delta MAE8.758us、最大17.761us。M768 n1/n2/n3/n4的delta误差-7.259/-8.977/-3.992/-12.378us，
M1205/n3为+17.761us。n3代理统一预测86.956，但M529/M768/M1205实测80.98/82.94/81.84，
资源代理本身高估约4..6；M768/n1代理只偏约-0.58，却仍低估减速7.26us，表明不能只修代理
一个分项来宣布敏感度正确。独立T0高估与竞争低估抵消在首场PMU中仍存在。
第二场继续原脚本运行，保留分层误差；后续将结合完整时间线影响评估候选价值，不用微小总体
百分比误差替代真实计划误差和选择效果验证。


## Gather队列留出第二场评分完成：压力与敏感度误差不能合并处理

第二场session2.jsonl及日志已从Arm-codex-internal同目录收回。运行
`.venv/bin/python tmp/joint_cost_model_20260911/gather_queue_holdout/evaluate.py session2`，
complete/grid、seed、冻结model SHA、数值/CPU和64事件有效性检查通过；未修改参数。
协议沿用本节前述8T、H4096/F512、NUMA3、32输入/输出槽、M12/W13 B32背景、
5warmup+31正式、seeds612221/612222，非真实瞬时MoE调度。

第二场active service MAE11.197us/MAPE2.319%，paired delta MAE7.084us，最大18.614us。
M529/M768/M1205 n0的基线误差分别+3.877/+9.571/+17.638us。
M768 n1/n2/n3/n4增量误差-3.919/-4.897/+3.510/-8.624us；M1205/n3为+18.614us。
M768/n4总service仅高1.000us，仍是基线高估和竞争低估抵消；三场不支持只按总MAPE采用。

增加只读条件诊断，结果保留`gather_queue_holdout/measured_pressure_diagnostic.json`：
对每个M的同轮n0，计算实际占用代理差，再以冻结T0与k计算
`delta_conditioned=median_round(T0*k*(q_active-q_own_n0))`。
这是使用待预测样本实测压力的回顾诊断，不是planner输入，也不重新标定q0或敏感度。
两PMU场delta MAE从8.758/7.084us变为8.481/4.545us。
M1205/n3误差从+17.761/+18.614us降为-3.870/-2.412us，说明该点的资源压力预测贡献较大。
但M768/n2误差从-8.977/-4.897us变为-14.401/-9.786us；
M768/n3变为-12.683/-5.454us。所有12个条件诊断点均低估增量。
因此不能只修q(n)就宣称前台敏感度已经正确；有效代理与延迟映射、状态迁移仍有残差。
这不是唯一硬件因果归因，且稳态槽轮换与真实路径T0的协议差异保留。

决策：冻结当前候选作为受限稳态对照，不把count多项式直接用于真实异构阶段。
下一步完整时间线适配应分别暴露独立服务、需求、响应与交接项；先对现有真实trace做
适用域与关键路径覆盖检查，缺失状态必须显式记录。未见完整计划、混合宽度和等预算搜索
仍未验收，默认planner保持不变。本次未启动新基准，也未更改生产或模型参数。


## 完整anchor时间线审计：当前完成路径全部落在16T lane

新增独立Lab `full_timeline_audit/extract.py`，从queues_runner分析器快照扩展，
仍验证原始frontier/session/trace身份、配对、数值、所有任务envelope和完整调用序列；
只保留指定计划（默认anchor），但保留该计划全部expert。没有修改frontier或旧compact。
原始state_order_holdout两场trace各663MiB留在Arm-codex-internal同目录；离线各验证533调用，
提取31个anchor正式pair的全部224expert至full_timeline_audit/session1.json与session2.json。
这是既有真实route、8/16T实际full stripes、NUMA3 CPU240..319、BF16/SVE256、H4096/F512、
Ntile16、4副本/216MiB scrub/固定workspace/early_merge off；seeds611681/611682。
没有启动新基准。入口、gather、W13、W2与最终merge继续区分。

`analyze.py`核对224expert集合、拓扑依赖、阶段顺序、core_begin、31pair，以及旧compact保留
的全部15expert逐字段精确相同。沿最后W2任务反向选择实际最晚完成的直接依赖，逐样本得到
观测依赖路径，服务时长+初始入口/交接残差严格还原最后W2端点。该分解不是硬件因果证明，
也不是自主预测；分项中位数不保证可加。

两场62个pair的最后expert均为167，路径始终为CPU240..255的42个16T任务，
包括expert96/M1718、230/M77、180/M73以及39个M<=60任务。
全部expert计算完成中位数28697.71/28642.22us；路径16T大M服务中位数18785.75/18771.72us，
小M服务9553.67/9458.48us，初始/依赖间隙272.85/319.35us。
次晚lane（core_begin56，8T）各自完成中位数28315.81/28256.70us，差约382/386us；
这不是逐pair最晚次序或可保证的加速余量。
路径gather/W13/W2服务和的中位数分别1311.86/18168.90/8800.02us与
1320.30/18021.60/8804.74us。真实完整计划的终点不能由当前15expert前缀或仅8T精度替代。

`baseline_transfer.py`仅在已有16T独立采样存在完全相同M时做描述性匹配，不插值：
覆盖42个路径任务中的29个，缺13任务/10种M（14/16/18/20/22/26/32/43/73/77）。
使用各独立会话中位数的中位数作参照；独立为expert96/CPU304..319，anchor为CPU240..255，
除expert96外route/expert也不同，因此差值不能唯一归因于外部竞争。
已匹配任务W13差值之和+318.35/+252.69us，W2+324.405/+332.885us；这也是分项中位数之和，
不是全计划预测误差。expert96/M1718单独gather从独立参照约488.14us到联合957.67/958.22us，
差+469.53/+470.08us；W13差+111.56/+83.48us、W2+47.06/+42.51us。
后续多个小M的GEMM也比独立参照慢约30..38us。小M独立gather包络含初始worker到达散布，
直接相加反而高估后续gather，说明必须保留worker服务与入口分离，不能使用独立包络查表。

结果保存在full_timeline_audit/results.json及baseline_transfer.json。运行命令：
`.venv/bin/python tmp/joint_cost_model_20260911/full_timeline_audit/analyze.py` 和同目录
`baseline_transfer.py`；两脚本Ruff通过，完整数据/旧compact一致性与依赖路径恒等式通过。
独立extract脚本Ruff通过，两场完整trace解析成功；输出独占创建，不覆盖旧证据。

下一优先级据此调整为16T完成路径闭合：先用worker级服务处理大M gather与入口，
补10个实际缺失M的完整真实路径基线，再检验后续小M的响应/状态迁移；8T需求与双向竞争
仍保留在联合模拟范围，因为它们会影响16T服务，不能把其他lane删除。
当前dynamic_dag接口硬限8T且在prepare中一次性结算gather；dynamic_stage调用小M基线，
每阶段仅首块请求窗口有共享需求。unified_gather_candidate虽有gather事件，却仅支持无共享核
的root cohort且GEMM不减速。两者均不能只换基线函数就宣称完整模型，需合并动态gather、
完整依赖/CPU释放和双向阶段服务。未更改默认planner，完整目标仍待验证。

本次几何沿用(8/16,0,0,1,1)：W13/W2整阶段8/4MiB，8T owner1MiB/512KiB，16T owner512/256KiB。
提取器与旧版的diff仅涉及保留计划过滤和全部task输出，原有验证路径未删改；分析器逐pair
还原依赖路径并与旧compact精确对照。静态审查无阻塞项；未做新的自主预测或性能改善声明。


## 16T关键路径gather worker分解与实际缺失形状补采启动

沿用full_timeline_audit/extract.py的--worker-details，从两场已有trace各完整校验533调用，
提取全部anchor worker到session1_workers.json/session2_workers.json；没有修改原trace。
`gather16_workers.py`核对31pair、完整tasks与上轮输出一致、16worker CPU顺序和包络恒等式。
expert96/M1718的联合gather两场envelope957.67/958.22us，平均worker服务858.943/857.857us，
入口散布60.66/61.89us；旧独立CPU304..319的envelope486.70/489.58us，
平均worker服务412.087/418.602us，入口散布64.11/66.54us。
以各场独立local-worker服务中位数替换联合服务，保留联合实测worker起点，预测包络
484.43/491.06us，完成残差中位数473.29/462.53us（MAD7.16/15.50us）。
因此入口错位不能解释大部分差异；差异发生在worker服务区间内，但CPU位置、状态与
外部竞争仍有混杂，不能直接认定某个内存机制。实测起点仅用于诊断，不供自主planner使用。
结果gather16_workers.json，脚本Ruff与完整数据检查通过。

新增prepare_width16_anchor_missing.py，独立生成width16_anchor_missing：expert96原M1718的
真实route前缀，M14/16/18/20/22/26/32/43/73/77十个缺失形状，加M12/M1718控制及anchor，
共13plans。目标改到原anchor CPU240..255，16T实际full stripes(16,0,0,1,1)，W13/W2整阶段
8/4MiB、owner512/256KiB、H4096/F512、BF16/SVE256、Ntile16、NUMA3、80worker，
4权重副本/216MiB scrub/固定pretouched workspace/early_merge off不变。
控制点用于与旧CPU304..319数据桥接；新expert前缀不是其他expert相同M的相同route数据。
这是基线标定与位置迁移检查，不是未见形状泛化验收，没有新模型预测或拟合。

资源/拓扑检查、准备脚本Ruff和run.sh shell语法检查通过；远端扩展与profile SHA分别仍为
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`和
`cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41`。
确认旧基准进程不存在、两场离线提取结束后，执行
`bash tmp/joint_cost_model_20260911/width16_anchor_missing/run.sh`，seeds612241/612242，
每场5warmup+31pair，严格串行。exec6548当前活跃，尚无正式评分结果；不因观察等待重启。
新增analyze.py逐形状分开保存phase、mean-worker、arrival spread与16worker服务，已通过Ruff。
生产/默认planner和冻结模型参数不变。


第一场随后完整校验通过并收回，`width16_anchor_missing/analyze.py 1`通过frontier/seed、
31pair、CPU、phase/worker包络一致性检查，analysis_session1.json保存全部12形状。
同CPU240..255独立M1718 gather phase486.14us、mean-worker416.831us、arrival spread61.09us，
与旧CPU304..319独立数据接近。W13/W2 phase10817.04/5223.99us；M12为92.06/46.53us。
新增缺失M的W13/W2 phase中位数（us）：M14 163.18/69.84，M16 165.87/71.06，
M18 169.70/74.78，M20 173.48/75.95，M22 189.83/81.74，M26 250.06/104.99，
M32 256.34/111.43，M43 335.98/150.31，M73 562.92/245.05，M77 570.77/258.18。
这些是独立首任务，不能直接忽略后续任务状态与竞争后相加。

同一新会话的anchor与iso_m1718逐pair对照，已核对两边16worker均CPU240..255，
同expert96、M1718与几何，原route不变。联合减独立gather phase增量中位数472.76us
（MAD13.79），mean-worker增量446.020us（MAD10.121），arrival spread差-3.74us（MAD8.46）。
W13 phase增量97.75us（MAD18.41）、W2增量39.09us（MAD23.04）；结果anchor_pair_session1.json。
这比跨位置比较更直接支持联合环境下gather服务显著减速，仍不唯一识别LLC/DRAM机制。
第二场继续exec6548原脚本；等待重复性后再进入模型标定，无参数更新。


## 16T缺失形状两场完成；完整224任务零竞争对照首次闭合

exec6548返回session2 validated/COMPLETE/exit0，两场共1066调用完成数值及trace校验。
第二场结果收回后运行`width16_anchor_missing/analyze.py 2`，形状、seed、CPU、worker/phase
一致性通过。第一场中位数预测第二场，同形状W13 MAE2.458us/MAPE0.765%，
W2 1.453us/0.827%，gather1.155us/1.339%；gather最大相对差为M77的7.179%。
这是重复性检查，非新形状留出。第二场同核anchor-iso M1718配对gather envelope增量
467.41us（MAD14.06）、平均worker增量438.919us（MAD11.834）、arrival spread差+5.47us；
W13/W2 phase增量95.04/30.18us（MAD14.54/15.88），anchor_pair_session2.json保留逐pair值。
与第一场重复支持联合服务减速；两场均已结束，无基准后台继续运行。

新增独立Lab full_baseline_candidate，仅作为完整计划零竞争消融，不替换默认planner：
- baseline.py：8T使用冻结large_model_v2与gather多panel服务；16T仅使用已采相同M的
  session1 GEMM阶段中位数与16个local-worker gather服务中位数，不使用gather入口包络。
  取值优先新anchor位置批次，其次旧first_rows/bulk/h1/h2/h4；各条目保留原CPU/source。
  未测16T M明确拒绝，不插值、不假称任意形状泛化，其他width也明确拒绝。
- simulate.py：包含全部224expert、12288route、真实CPU/依赖/通知顺序；gather按worker
  初始入口与CPU可用时刻计算，随后W13/W2与交接。无任何竞争增量，无实际trace端点输入。
  初始入口使用既有family_small_m的31个完整相关场景；setup2.28us/gap0.39us和
  publication的base2.758062/per-successor0.196938us沿用8T标定，16T迁移尚未单独验收。
- replay.py：完整原anchor回顾评分，实际trace只用于评分，不参与当前参数拟合。

运行`.venv/bin/python tmp/joint_cost_model_20260911/full_baseline_candidate/baseline.py`、
同目录`check.py`、`replay.py`。完成预测中位数27148.855us；原两场真实trace完成
28697.71/28642.22us，误差-1548.855/-1493.365us（-5.397%/-5.214%）。
九条lane均低估，8T各lane误差约0.83..2.42ms，因此不能仅补16T后忽略其他lane。
结果含全部任务分阶段误差、31个自主入口场景的完整时间线、所有lane端点。
这是不同组件拼合后的回顾零响应对照，不是完整新模型，也不满足联合完成验收。

对16T lane做均值可加分解（预测减实测，us）：
| 分项 | 第一场 | 第二场 |
|---|---:|---:|
| gather包络和 | -612.472 | -620.026 |
| W13阶段和 | -365.013 | -242.517 |
| W2阶段和 | -498.624 | -535.105 |
| 初始入口及所有阶段外间隙 | -60.772 | -119.088 |

每场逐lane校验分项和严格等于均值端点误差；这些数不与中位数端点误差强行相加。
说明只加M1718 gather修正仍不足，后续GEMM状态/竞争响应也需进入模型。
下一步在这个完整DAG接口上接可独立消融的动态gather和GEMM需求响应，保留零响应对照；
不得把全计划残差直接拟合为宽度倍率或经验奖励，也不能由本对照声明可靠泛化。

验证：单任务、独立lane、依赖/CPU释放、无序共享CPU拒绝、NaN拒绝检查通过；
224个provider输出及31个全DAG场景在格式化后精确复算一致，未支持M/width拒绝通过。
初次Ruff报告7处单行分号/冒号风格问题，格式化后全通过；一次域外检查误将已测M15作为
非法点，改为确实未测M38后通过，没有修改provider适用域或模型参数。identity.json记录
当前源码、模型、输入profile/frontier身份；results.json和decomposition.json保留完整结果。
本轮仅Lab Python/产物/模型文档，未改Production ABI/kernel/build/default，无提交。


## 完整DAG动态gather/GEMM事件与双向请求反馈消融

新增独立Lab full_dynamic_candidate：所有224expert的worker gather、W13/W2请求窗口、
剩余private阶段、通知与CPU释放均在同一预测时间线上推进，不使用待评分trace的实测起点。
服务基线/31历史入口场景/setup/gap/publication保持full_baseline_candidate不变。
同时存在的请求窗口名义速率d，以lambda=d*v形成实际请求速率；
每个接收任务速度v=1/(1+sum_type(a_type_to_receiver*sum_other_expert(lambda)))。
同expert内部worker请求不重复计入外部压力；采用0.5阻尼固定点，容差1e-10，最大256轮，
不收敛报错，不能偷偷降级成静态请求。当前四组最多32轮收敛。

这是接口/机制消融，需求与敏感度仍有明确未标定部分：
- gather需求仅逻辑read+write KiB除以独立worker服务，无LLC/DRAM层级或读写独立映射。
- GEMM每阶段仅首块窗口有Q13=8192KiB/Q2=4096KiB，后续private不发共享请求。
  该假设不覆盖已测大M持续流量和状态变化，不能作为最终需求模型。
- gather<-gather系数0.0016、gather<-GEMM0.0002沿用早期8T split失败诊断；
  GEMM<-gather/GEMM使用0.08/(200000/1024)，200GB/s仅归一化尺度，不施加硬带宽上限。
  向16T和异构请求的迁移未经标定；没有依据本批全计划残差拟合新系数。

`check.py`通过单任务隔离身份、双对称gather的解析反馈解（速度满足v=1/(1+v)）、
依赖通知/CPU释放及无序共享CPU拒绝。`replay.py`首先在31个历史入口场景上逐个核对
224任务所有阶段端点，zero与旧完整DAG差<1e-7us；随后四组完整回放均正常结束，Ruff通过。
此次仅本地离线预测，没有远端新实验；源与输入身份在identity.json。

| 响应开关 | 完成预测/us | 第一场误差/% | 第二场误差/% |
|---|---:|---:|---:|
| 全关闭 | 27148.855 | -5.397 | -5.214 |
| 仅gather接收响应 | 27218.159 | -5.156 | -4.972 |
| 仅GEMM接收响应 | 27414.505 | -4.471 | -4.286 |
| 两者均开启 | 27477.503 | -4.252 | -4.066 |

完整时间线推进已接通，但四组都未通过既定完成误差门槛，不采用为最终模型。
M1718/gather预测增量仅62.16us（both）对比独立同核新实测增量约467..473us；
W13预测增量11.47us、W2为0，也明显不足以表达新同核配对95..98/30..39us的响应。
这些是跨会话诊断量级对照，不是新预测的前瞻评分。不能只按总完成较接近就认定响应正确。
results.json保存四组所有场景/事件/系数/分数，ablation_summary.json保存expert96分阶段增量。

下一步保留该事件引擎和zero回归对照，把硬编码首块/零后续需求替换为可分段的状态需求输入，
并分别校准前台敏感度；不能将参数简单放大至全计划正确，也不能把稳态count曲线直接应用
于异构瞬时请求。16T未测M、完整竞争泛化、候选选择和等预算搜索仍待完成，默认planner不变。


## 显式分段需求接口完成：冻结成本、读写请求与敏感度分开

新增独立Lab segmented_dynamic_candidate，由上一版完整动态引擎建立，未改旧快照。
每个W13/W2阶段现在显式接收segments：state、service_us、read_kib、write_kib、sensitivity。
所有service_us必须有限且正；读写请求与敏感度必须有限非负；阶段service之和必须等于冻结T0。
每段结束才进入下一状态，阶段结束再发布后继；无实测阶段端点参与推进。
请求按当前段进度发出，积分read/write分别与输入请求量核对；零请求可以显式表示private段，
未测状态不自动生成零请求。当前响应仍将read+write按gather/GEMM issuer类型合并，
并非已有LLC/DRAM或读写独立排队模型；分段敏感度独立于发出请求量。

segments.py把旧首块请求/后续private假设转成显式两段：旧request段sensitivity1、
private段sensitivity0；私有服务不因新增接口自动变成共享带宽工作。
`check.py`通过隔离成本保持、读写守恒、同速率段拆分不改变双任务事件、负需求与成本不守恒拒绝。
`replay_legacy.py`对四组系数×31历史入口场景执行全部224expert回放，逐任务逐阶段与旧输出比对：
zero/gather_only/gemm_only/both最大端点误差2.692e-10/7.640e-11/2.765e-10/1.164e-10us。
所有运行的read/write请求守恒校验通过，固定点最多32轮。exec87755正常exit0，Ruff通过。
这是实现等价验证，不是预测精度提升；完整模型仍有上一版4%以上的完成误差。

`demand_inventory.py`只读旧eight_stage_demand_20260910/analysis.json，保存两会话平均的
每调用read/write预算、M/阶段/copies/stride/状态，不从GB/s直接充当运行时请求率。
旧表仅8T/M1,12,13,48；与完整anchor仅29/224任务、225/12288路由行形状重合。
这是形状重合上限，不是可信预测覆盖率：旧持续B1/B32协议与真实四副本/scrub状态尚未桥接，
故runtime_state_matched_tasks明确记0；within-stage temporal_segments明确记null/unmeasured。
不能从整阶段M差分得到物理panel请求，也不能把未知状态当零需求。
B32/stride1整阶段读预算M12 W13/W2约7.922/3.486MiB，M48约16.054/7.341MiB，
再次表明统一一次Q8/4MiB不足。该连续探针的数据不能直接替换真实路径阶段预算。

产物legacy_jobs.json、legacy_parity.json、demand_inventory.json、identity.json保留在新目录。
命令为该目录check.py/replay_legacy.py/demand_inventory.py，旧source和输出均未覆盖。
接下来应补16T/大M请求预算及时间分布并检查协议迁移，再经分段接口进入完整动态模型；
不会只把原固定Q切成更多块或用全计划残差放大系数。此次无远端新采集、无生产/默认planner变化。


## 16T大M整阶段需求探针：首次数值烟测失败，尾块A填充修复中

新增large16_stage_demand，复用独立stage_async JIT调用和PMU协议，限制一个16T team，
CPU304..319、controller240、NUMA3，M12/48/1718、W13/W2、B1/B32，加idle共13格。
实际(16,0,0,1,1)、Ntile16、W13/W2整阶段8/4MiB、owner512/256KiB、BF16/SVE256。
常数A/B、预分配输出、连续synthetic route，不是实际expert route或4-copy/scrub真实上下文。
每格64call lead-in后观测；M1718扩大到500ms，其余100ms。完成数改为落在窗口内的
完成边沿计数，避免一律丢弃跨窗口起点的调用；仍报告一调用量化占比，不将长阶段100ms
少量整调用计数视为精确吞吐。formal计划5warmup+31round，control及seeds612261/612262串行。

首次构建成功无警告，JIT源码SHA仍1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629；
native SHA9d07560b85e42430b17d73ff83d7dd9022639e366c2b8d0fba9a2c4a65c7e25f。
无PMU smoke在首格M1718/W13/B1失败：W13 numerical failure，exec3777已exit1，
未产生正式测量或任何成功格。原binary/native、smoke.jsonl/.stderr与stage_service_failed.cpp保留。
定位到扩展最大M时A只按逻辑1718行分配，而r2尾kernel读取按physical8行步幅打包的A panel；
原48行上限的smaller-tail格有余量，新最大值尾块没有。v2在A13/A2各加12行填充，
不改JIT/算子kernel，输出本来已留12行余量。build_v2.sh/native_v2/独立build_identity_v2保留版本。
当前exec38936构建v2，需重做全部数值与64事件烟测才可启动formal，不把失败数据用于拟合。

分析器analyze.py保存read/write idle-subtracted每调用预算、占用/命令代理、service、
PMU/control差、最少调用数和一调用量化比例，保留负idle差值不截断；未知段内时间分布不填零。
Python Ruff和shell语法通过。8T大M仍待本批结束后采集，不并行运行性能实验。


v2修复后无PMU13格和64事件PMU13格均complete，analyze.py分别检查grid、16T身份、
数值、CPU、完成数/窗口及PMU enabled/running有效性通过。构建无警告；
native_v2 SHA96e9772c570bbd9470252b552d1e2ec886733dda73ed00acbdd6b07e0c111fdd，
source SHA7178971fa3ecd2fa62e0b14f910e35b69958865cf3b43f2c1ba8fb3015ea0fae；JIT未变。
失败和成功烟测/构建身份已同时保留本地与远端。修复只涉及独立驱动A填充，无生产改动。
随后在通过PMU烟测校验的同一串行命令中启动run.sh，exec7252当前活跃：
control seed612261，再PMU session1/2 seeds612261/612262；共13×36×3=1404格。
尚无formal结果，不以单轮smoke作流量或性能结论。8T大M将在本批结束后处理。
本轮作为Lab测量M类变更，保留原无竞争成本和所有模型参数；Python Ruff、shell语法、
native构建、数值/route guard和PMU烟测通过。回退边界为large16_stage_demand独立目录。


## 16T需求控制场完成；8T同协议版本准备

exec7252仍为原串行run.sh；远端control.jsonl已complete，收回运行analyze.py control，
13格×36轮全grid、16T身份、数值、CPU、完成数/窗口检查通过，analysis_control.json保留。
控制服务中位数（B1/B32，us）：M12 W13 89.79/91.33、W2 50.81/52.16；
M48 W13 320.48/328.44、W2 164.89/169.45；M1718 W13 11017.04/11060.04、
W2 5450.18/5458.36。该计时包含探针team barrier，不替代真实路径阶段T0。
M1718窗口最少W13完成45次、W2完成91次，一调用量化比例最大2.222%/1.099%；
后续每调用请求预算必须保留此限制，不声称亚百分比绝对流量精度。
第一场PMU已进入正式轮次，第二场仍排在其后，原进程继续，不重启。

本地新增large8_stage_demand，为匹配协议的8T后续采集准备：M12/48/1205，W13/W2，
B1/B32，CPU312..319、NUMA3、controller240；(8,0,0,1,1)，stage8/4MiB、owner1MiB/512KiB。
保留16T探针相同1718行A/输出容量、1720 route保护范围和清理footprint，已有packed-A尾填充修复。
仅改变执行宽度和最大被测M，M1205窗口500ms，其余100ms；拟seeds612281/612282。
C++精确diff、Python Ruff、shell语法、protocol几何/CPU/窗口与driver校验一致性检查通过。
当前仅本地准备，尚未构建、烟测、同步或正式运行；待16T三场原进程结束后顺序启动。
无模型参数或默认planner变更，当前没有请求量结论，整阶段预算与段内分布仍需区分。


## 16T需求首场PMU完成：大M在B复用状态下仍有非零共享流量

exec7252输出session1 complete，第二场继续原脚本。session1.jsonl及日志已收回，
`large16_stage_demand/analyze.py session1`通过完整13格/31正式+5warmup、16T/CPU/数值、
64事件coverage和running/enabled>=0.99检查，二进制与无PMU控制相同。
`compare.py 1`进一步核对seed612261、31轮配对，按同轮idle扣除的每调用请求预算计算
B32-B1配对增量，contrasts_session1.json保存逐pair值；Python Ruff通过。

首场每调用idle-subtracted读/写预算（MiB），不是逻辑字节或实际单次调用的直接地址归因：
| M | 阶段 | B1读/写 | B32读/写 |
|---|---|---:|---:|
| 12 | W13 | 0.0127/0.0005 | 6.1482/0.0014 |
| 12 | W2 | 0.0129/0.0005 | 2.7596/0.0013 |
| 48 | W13 | 0.0654/0.0002 | 12.8201/0.0050 |
| 48 | W2 | 0.0132/0.0005 | 5.5648/0.0122 |
| 1718 | W13 | 27.1229/-0.0066 | 49.3853/0.2260 |
| 1718 | W2 | 6.5725/6.8140 | 17.9561/9.5132 |

B1小M接近流量底噪，但大M仍有读写请求；不能把重复B解释为整个阶段零DRAM需求。
B32大M也远非一次Q8/4MiB，且W2写流量不可省略。32副本仅是访问协议，不等同所有行完全冷。
M1718/W13 B1写预算-0.0066MiB是idle扣除后的近零负估计，原值保留，不能作为负物理请求
输入模拟器，也不直接截断后声称精确零；需要重复性/测量底噪处理。

同轮B32-B1配对read增量：M12 W13/W2 6.1357/2.7465MiB，M48 12.7590/5.5540MiB，
M1718 21.4757/11.3347MiB；大M W2配对write增量2.6770MiB。
配对中位数不要求等于两个边际中位数相减，变化也不唯一等于B物理流量（A/output状态可变）。
PMU计时相对控制的最大绝对变化0.2832%，M1718最少完成数W13/W2仍45/91。
这些是第一场初步结果，不是跨会话稳定性或真实4-copy/scrub路径的迁移验证。
当前未拟合新参数、不将整阶段预算随意分到各panel；等待第二场，再顺序启动8T版本。


## 16T需求两场完成：W13读预算接近，W2存在稳定的跨会话偏移

exec7252输出session2 complete/COMPLETE/exit0，control+两PMU共1404格完成，第二场收回后
analyze.py session2、compare.py 2通过完整grid、计数器、数值/CPU、二进制与seed检查。
两场PMU计时相对控制最大绝对偏差均约0.283%；不能据时长稳定推断流量稳定。
M1718的两场整阶段idle-subtracted预算（MiB/call）：
| 状态/阶段 | 第一场读/写 | 第二场读/写 |
|---|---:|---:|
| B1/W13 | 27.1229/-0.0066 | 26.4368/-0.0071 |
| B32/W13 | 49.3853/0.2260 | 48.4742/0.2747 |
| B1/W2 | 6.5725/6.8140 | 4.8348/4.5230 |
| B32/W2 | 17.9561/9.5132 | 15.2858/6.9027 |

W13读预算变化约-2.53%/-1.84%，W2 B1读/写变化-26.44%/-33.62%，B32 -14.87%/-27.44%，
明显大于W2一调用计数比例1.10%。同轮B32-B1的W2 read增量11.3347→10.4912MiB、
write增量2.6770→2.3697MiB，相对更接近但也不能声称完全稳定。
repeat.json保留逐条件比较，近零量不强行计算百分比。

w2_drift.json进一步按31轮分析（事后描述性诊断）：M1718/W2 B1 read场内MAD
0.061/0.048MiB、write0.112/0.057MiB；B32 read MAD0.120/0.085MiB、write0.086/0.061MiB。
前10/后10轮中位数接近，没有相当于跨场差异的单调漂移。说明不是简单场内随机计数波动，
但当前证据不唯一识别物理分配、缓存驻留或其他共享状态原因。大M非零流量的现象重复，
不能把两场绝对值平均后作为固定、已泛化的需求参数。模型仍保留unknown状态和测量范围。

确认16T终止后同步并构建large8_stage_demand，JIT SHA不变；native SHA
80c46a4ec0ab18ffd7a619ed849f69bef104e201ba8b3c0f59caa0e6e4612d59，source SHA
bdde4b8bc33be00eb762ca93a49656cab79bbbc00b76e1cf7a2725706e451991。
13格无PMU烟测已complete并通过analyze.py smoke，PMU烟测exec56834进行中。
CPU312..319、NUMA3、(8,0,0,1,1)、stage8/4MiB、owner1MiB/512KiB，M12/48/1205，
其余与16T相同，正式采集尚未启动。不将16T跨场偏移悄悄拟合进宽度倍率，默认planner不变。


8T PMU烟测随后正常结束并通过64事件/全grid检查，原烟测与构建身份已取回本地。
在同一串行命令中启动large8_stage_demand/run.sh，exec26969当前活跃，control612281后
PMU612281/612282，13格×36轮×3，M1205窗口500ms。16T已完全终止，未并行运行基准。
当前8T无formal结果，不使用smoke作请求量结论。参数和默认planner不变。


## 16T请求分布与接收敏感度分离诊断：整段响应不能直接采用

新增request_distribution_diagnostic/run.py，仅修改完整anchor的expert96/M1718，其他223任务
维持已有首窗口/零后续需求。固定该任务无竞争T0，以large16_stage_demand两场B32 read/write
整阶段预算分别作输入，比较前半段发出/全阶段均匀/后半段发出三种假设；均为未测时间分布，
不是物理上下界或真实路径迁移结论。另独立比较仅首窗口敏感/整个阶段敏感，两组都使用冻结
旧响应系数，不拟合full-plan残差。段边界包含首窗口与半阶段，改变发出分布时保持接收mask不变。
12组×31历史入口场景完整回放结束（exec86227 exit0），stage成本及read/write请求守恒检查通过。

在该受限对照中，三种分布的完成时间跨度仅0.101..0.144us；两场预算切换也影响很小。
这不证明其他任务/时间尺度的分布不重要，而是当前只有一个长16T任务换需求、其他背景仍旧模型。
仅首窗口接收时，均匀分布完成27477.900/27477.878us，接近原both27477.503us。
整个阶段接收时，完成27885.020/27884.965us，比原both增加约407.5us，但不能据总误差变小采用：
score.py对实际两场trace分项检查，M1718/gather仍低估408.1/408.7us；
W13从低估89.1/61.0us变成高估259.4/287.5us，W2从低估46.0/41.5us变成高估34.2/38.8us。
总完成虽更近，明显包含分项抵消。需要标定分阶段/输入/历史的接收敏感度，不能把整个阶段
都乘同一响应，也不能将单点总时间修好视为可靠泛化。

结果results.json/scores.json/identity.json保留全部假设和31场景预测，Python Ruff通过。
来源是已看过的数据与假设分布，属于回顾敏感性诊断，不更新正式模型参数或默认planner。
下一步优先独立约束接收敏感度，并继续需求状态/真实路径迁移验证；总预算和段内分布问题仍未关闭。

同轮8T控制场已complete并收回，large8_stage_demand/analyze.py control通过全grid/数值/CPU/
窗口检查。M1205 B1/B32服务中位数W13 15461.40/15489.12us，W2 7623.30/7643.02us；
500ms窗口最少完成32/65次，一调用比例3.125%/1.538%。这些是探针含同步时长，非真实T0。
exec26969仍运行第一场PMU（最近确认到23/31轮），后续第二场保持原脚本；未重启或并行采集。


## 8T大M需求两场完成；16T真实路径接收响应标定启动

exec26969返回session2 complete/COMPLETE/exit0。8T control+两PMU共1404格全部结束，
两场analyze.py及compare.py通过完整grid、数值/CPU、64事件、binary与seed校验。
PMU计时相对控制最大偏差第一场0.4545%、第二场0.3349%。M1205整阶段预算（MiB/call）：
| 状态/阶段 | 第一场读/写 | 第二场读/写 |
|---|---:|---:|
| B1/W13 | 13.3563/-0.0084 | 12.7065/-0.0103 |
| B32/W13 | 39.5745/0.5889 | 39.5503/0.5918 |
| B1/W2 | 2.8949/3.5999 | 3.6787/4.7583 |
| B32/W2 | 16.0585/8.1993 | 17.1721/8.9449 |

B32 W13读预算接近，但W2仍有跨场变化；B1 W2变化更大，说明此前16T状态差异不是仅16T才有。
M1205同轮B32-B1 read增量W13 25.9357/27.1739MiB、W2 13.2522/13.3292MiB，
W2 write增量4.7887/4.2140MiB。单份权重依然有大M流量，不能视为总需求为零；
整阶段预算不直接归因于A/B/output或任意panel。近零负写估计保留，不输入为负物理请求。
所有原始与analysis_session1/2、contrasts_session1/2结果保存large8_stage_demand。

新增prepare_width16_response.py和width16_response，回到真实4-copy/scrub路径做响应标定，
不是新模型泛化留出：prefix expert230/M77和target expert96均16T CPU240..255，
target M12/48/1718使用同expert真实route前缀；背景expert85/1205、14/768、115/714各8T。
每个M比较none、local1、local3、cross3，共12实验+原anchor=13plans；
local背景CPU256..279，cross背景CPU280..303。前缀使背景有时间进入GEMM，
但实际W13/W2/gather重叠必须由trace检验，不能仅由准备意图认定。

源route SHA、layer4、前缀和背景实际M、所有变体top-k/active expert集合及背景计数已核对；
桥接重排保留其余任务依赖并传递约简，所有共享CPU祖先关系通过validate_resources。
其余expert等待整个selected cohort完成；inactive背景在target完成后运行。
16T目标full stripes owner512/256KiB，8T背景owner1MiB/512KiB；stage8/4MiB、
H4096/F512/BF16/SVE256/Ntile16/NUMA3/80worker、early_merge off、固定workspace保持。
新增前缀本身会受竞争，配对目标duration差不能脱离实际时序直接当纯敏感度参数。

准备脚本Ruff、run.sh语法与资源/路由验证通过。确认8T探针终止、扩展/profile SHA仍为
原dd554e.../cdccff...后，执行`bash tmp/joint_cost_model_20260911/width16_response/run.sh`，
seeds612301/612302、5warmup+31pair、两场串行；exec59834当前活跃，尚无新实测结果。
新增analyze.py预备分别输出foreground三阶段配对增量/MAD、prefix时长变化和逐背景阶段重叠；
Ruff通过，实际数据验证待采集结束。无新参数拟合或Production/default变更。


## 长背景真实16T响应两场完成：阶段重叠存在，但响应较弱

exec59834返回session2 validated/COMPLETE/exit0，两场各533调用完成数值/实际M/trace校验。
两场compact已收回，analyze.py 1/2与overlap_session1/2.json分开记录目标、前缀和背景时序。
同LLC三背景的前台配对增量中位数（W13/W2，us）：M12 1.19/0.13与2.76/0.31；
M48 0.16/0.20与1.10/-0.45；M1718 12.23/8.84与17.78/8.51。
第一场M1718两阶段增量MAD14.70/15.61us、第二场7.94/7.66us，弱响应不应拟合成强确定性规律。
M1718同域gather差-8.83/-10.28us，跨域-12.85/-11.77us，也不能把所有外部环境影响
一概限定为正减速后直接覆盖固定T0；共享输入/状态及竞争应分开验证。

前缀达到了让背景进入GEMM的目的：两场M12/M48前台W13/W2在local3条件下，三个背景
的W13重叠比例中位数均约100%，gather约0。第一场前台M12/W13启动时，背景W13已运行
约195/443/460us；M1718/W13启动时约594/847/873us，并在目标期间部分转入W2。
这些只是实际阶段端点和已运行时间，不是缓存命中或B热的直接测量。
保留本批作为实际长背景低响应对照；它与完整anchor并不具备相同并发数和阶段请求变化，
不能据此否定anchor减速，也不能从近零增量辨识强响应系数。

新增prepare_width16_wave_response.py及width16_wave_response：固定原16T M77前缀和
M12/48/1718目标，仍比较none/local1/local3/cross3；背景改为三条21expert队列，
每条10个小M→一个长M→10个小M，长M为expert14/768、115/714、59/529，
其余60个不同expert按(M,id)排序选取并round-robin分配，实际M1..6，权重/route均来自原输入。
全部63背景与前缀/目标合计65expert；不重复虚构同一个expert，不改变top-k总路由量。
选取ID>=14且排除目标/前缀/长任务，逐个nested变体证明背景和前缀计数保持原值。

每条队列依赖前项；inactive队列首项等待target完成，其余非cohort任务等待整个cohort。
原任务祖先通过bypass保留并传递约简，所有共享核排序检查通过。前台CPU240..255，
local三lane CPU256..279、cross CPU280..303；真实16T/8T full stripes、stage8/4MiB、
owner512/256KiB及1MiB/512KiB、NUMA3/80worker/H4096/F512/BF16/SVE256/Ntile16、
4副本/216MiB scrub/固定workspace/early_merge off保持。
小/长/小队列用于创造不同阶段请求变化，实际是否覆盖目标W13/W2仍须trace验证。

准备器Ruff、run.sh语法、route SHA/计数、63背景唯一性/每队列21任务、资源祖先检查通过。
确认旧实验结束后执行`bash tmp/joint_cost_model_20260911/width16_wave_response/run.sh`，
seeds612321/612322、5warmup+31pair、13plans，两场串行，exec13366当前活跃。
复用同一目标响应/背景重叠分析器，正式数据验证尚待完成。这批仍是标定/诊断，不是泛化留出；
没有更新模型参数或Production/default，也没有并行性能实验。


## 小/长/小队列两场完成；预划分接收拟合未通过留出

exec13366返回session2 validated/COMPLETE/exit0，两场共1066调用完整校验通过，65expert
compact均已收回，analyze.py 1/2通过身份、seed、实际M/CPU、前缀先于目标和31pair检查。
local3配对目标增量（第一场/第二场，us）：M12 W13 16.68/12.74、W2 6.14/7.43；
M48 W13 9.62/9.68、W2 6.87/7.01；M1718 gather196.13/189.87、W13 15.34/19.32、
W2 29.94/49.08。跨域M1718 gather192.86/183.04、W2 -3.47/3.67。
相比单长任务背景，新队列产生更强请求变化，尤其大M gather和W2的local/cross差异。

在读取新队列duration前写入receiver16_candidate/design.json：仅session1的M12/local1、
M48/local1及对应n0用于拟合，M1718和local3/cross3及全部session2不参与拟合。
固定无竞争bank、prefix时序模型和旧首请求Q8/4MiB；8T小行背景使用已有0.632/0.458密度
响应，16T目标单独拟合W13/W2 first/reuse四个有效接收系数，非物理计算/访存比例。
训练采用冻结median入口场景，0..8粗细网格顺序拟合；评估使用全部31历史入口场景，
无实际起点进入自主预测。只保留会影响目标W2结束前的active队列/前缀/目标，
裁去inactive和cohort外后继不改变目标之前的通知/CPU释放；不据该子图评价完整计划完成。

拟合结果first/reuse：W13 0.90/0，W2 0.35/0。训练M48/W13配对增量-5.12us，
非负响应最低仍预测+2.01us，残差+7.13us；不能称标定成功或把零系数解释为物理不敏感。
参数扫描的预测跨度均非零，identifiability.json保留范围；不是因实现对参数完全无响应而落边界。
未参与拟合的7条件×2stage评分：
| 指标/us | 初始第一场 | 拟合第一场 | 初始第二场 | 拟合第二场 |
|---|---:|---:|---:|---:|
| 竞争增量MAE | 6.965 | 7.798 | 8.497 | 9.330 |
| 阶段绝对MAE | 9.791 | 10.450 | 10.020 | 10.592 |

因此拒绝采用本拟合，参数保持独立失败对照。前台W13起点也有误差：第一场M1718 local3/
cross3约提前205/198us，第二场250/245us，其中prefix仅贡献约20..40us，gather是主要余项。
不能通过这些起点误差反过来放宽留出或用实际起点充当planner输入。

conditional.py仅作诊断：实际前台stage起点、实际背景stage/worker起止与据此约束的请求窗口，
保持同一名义Q和拟合接收系数；未改变参数。由于使用实测时序，背景已包含真实反馈，
逐stage预测也可能越过实际后续事件，越界次数显式记录，不是可执行自主计划。
两场留出delta MAE为7.631/9.120us，与自主7.798/9.330us接近，不能靠修起点单独解决失败。
常量压力、跨边界、零敏感度积分检查通过；所有新增Python Ruff通过。

locality_exposure.json进一步检查M1718/W2：local3/cross3的名义首窗口请求重叠量，
第一场282.376/282.679MiB，第二场283.718/280.658MiB；小任务请求窗口重叠合计约3.9ms，
对应约47..48个背景小stage。它们并非一个有重叠、另一个没有；名义量不是PMU实测字节。
在近似相同重叠下，local W2明显减速而cross接近零；gather却在两种放置都明显减速。
这支持下一版区分跨LLC共享影响与本地共享域影响，并检查非线性压力响应；
不是LLC缺失/DRAM排队的唯一硬件因果证明，也不直接增加宽窄team奖惩倍率。

design/model/evaluation_session1/2、conditional_session1/2、identifiability、locality_exposure与
identity.json保留于receiver16_candidate。采前未冻结预测，属于预划分校准/留出及回顾诊断，
未宣称全模型泛化。所有远端实验已终止，本轮无Production/default变更，下一步从响应结构
改进继续，而不是继续用同一全局线性接收系数调数值。


## 本地共享域响应项接入与限定标定：跨域不变，尚待新条件验证

新增local_domain_candidate，保留segmented_dynamic_candidate原源码/输出。
每个team从CPU240..319映射到既有两个40核LLC分组，禁止team跨组；每段增加可选
非负local_sensitivity，默认0。请求进度和read/write守恒不变，响应为
`v=1/(1+s_global*P_global+s_local*P_same_domain)`，两种P均由当前其他expert实际请求率
及已有issuer系数组成。本地压力只是与LLC分组一致的有效共享域特征，不声称已测得LLC物理请求率。
gather本地敏感度保持0；本轮只给16T目标W2的first之后部分增加本地项，无team奖惩倍率。

check.py通过隔离任务/跨域恒等、同域双任务解析反馈解、本地项关闭和负系数拒绝。
replay_legacy.py执行四组×31场景完整224expert，local=0时端点最大差<3e-10us，
read/write守恒全部通过，exec4086正常结束，Ruff通过。

fit_local.py只用已查看的wave session1/M1718/local3 W2配对增量29.94us标定一个系数，
其他global first/reuse保持初始1/0（没有沿用上一轮失败的.90/.35），背景8T既有密度修正保持。
本地reuse系数0.28；median入口标定预测29.806us，31入口场景评估预测约29.754us。
这是事后结构校准，不是未见数据验收。
所有被评分条件的W13预测与local=0精确一致，cross条件和M12无reuse时W2也保持一致；
这验证局部项没有意外改变跨域或此前阶段，并不代表整体预测正确。

M1718/local3第一场增量误差由-29.94us变为-0.186us，第二场由-49.08变为-19.326us；
M48/local3 W2误差由-3.611/-3.751变为-1.269/-1.409us。
但M1718/local1由-0.35/-2.78变为+6.047/+3.617us，M48/local1也增加约0.92us误差。
排除标定条件后，其余8条件W2增量MAE初始2.375/2.883us，本地项2.910/2.810us；
绝对阶段MAE由3.827/4.057变为2.857/3.080us。不能只引用训练点或绝对误差改善宣称全面更准。
M1718/local1配对MAD8.22/14.85us，local3 MAD14.37/45.13us，应保留噪声尺度；
几us非训练条件差异不构成复杂曲线的充分证据，也不把本地系数当作精确物理量。

model/evaluation_session1/2/legacy_parity/identity.json保留完整结果。所有本轮计算均为本地
离线模型检查，无远端新采集、无Production/default变更。本地项保留为候选；下一步需新背景
数量/前台形状的冻结预测验证，并继续全局gather和小M响应缺口，最终完整planner验收仍未完成。


## 本地项前瞻验证已冻结并启动：两队列与M24/M96

新增prepare_width16_local_holdout.py，保持原三条21expert小/长/小队列、固定M77前缀及
16T目标expert96。目标形状M12/24/96/1718，各none/local2/cross2，加anchor共13plans。
M24/M96是接收响应的新前台形状（独立baseline已测），n2此前未参与本地系数标定。
原route SHA、nested背景/前缀计数、top-k/active集合、CPU共享祖先、实际full stripes检查通过；
其余expert仍等待整个selected cohort，未新增虚构任务或改变路由总数。

width16_local_holdout/freeze.py在采集前使用local_w2_reuse0.28，固定global first/reuse1/0、
原issuer系数、独立成本和31个历史入口场景，冻结12plans×2模型×31场景全部目标端点与阶段时长。
zero_local对照只关闭该本地项；W13/gather、cross条件、M12无reuse与none条件W2保持一致。
local2条件本地项给W2增加：M24约0.606us、M96约3.908us、M1718约17.139us，M12为0。
被冻结的是具体预测与参数，不根据新n0或新实测起点更新它们。frozen_predictions.json保存
输入/模型源码SHA；evaluate.py将核验这些身份，分别评分阶段绝对误差、对n0配对增量、
local2-cross2的W2配对差值和MAD。不会凭这批小实验自动切换完整模型或planner。

预测冻结、12计划/双模型/31场景完整性、准备器与评分器Ruff、run.sh语法均通过。
确认原扩展/profile SHA为dd554e.../cdccff...且无运行中的基准后，执行
`bash tmp/joint_cost_model_20260911/width16_local_holdout/run.sh`，seeds612341/612342，
5warmup+31pair两场串行，exec73116当前活跃，尚无新测量结果。
协议仍为Arm-codex-internal NUMA3/80workers/H4096/F512/BF16/SVE256/Ntile16，
前台CPU240..255、local背景CPU256..271、cross背景CPU280..295，
(16/8,0,0,1,1)、stage8/4MiB、owner512/256KiB与1MiB/512KiB，
4副本/216MiB scrub/固定workspace/early_merge off。没有Production/default或模型参数变化。


## 本地项两队列前瞻验证完成；回到完整anchor仍有约4%缺口

exec73116返回session2 validated/COMPLETE/exit0，两场各533调用数值/实际M/trace完整校验。
两场数据收回后，analyze.py 1/2、evaluate.py 1/2均通过前沿、seed、采前冻结source SHA及
31pair检查；没有以新n0或新实测起点重新预测，没有重新拟合0.28。
8个非control条件的W2阶段绝对MAE，zero_local→local为4.633→2.745us、5.843→3.813us。
四形状local2-cross2的W2差值MAE为3.360→3.182us、6.358→1.552us。

冻结局部差值预测与实测配对差（us）：
| M | local预测 | 第一场实测/MAD | 第二场实测/MAD |
|---|---:|---:|---:|
| 12 | -0.0028 | 1.49/1.17 | 2.89/1.30 |
| 24 | 0.6057 | 1.37/1.86 | 1.95/2.47 |
| 96 | 3.9064 | 1.41/3.02 | 2.69/2.35 |
| 1718 | 17.1547 | 9.18/17.30 | 17.91/22.45 |

M24有改善，M96少量高估；M1718第一场差值噪声较大，不能仅凭第二场贴近就称系数精确。
该结果支持把本地项保留用于后续组合实验，但只覆盖当前16T目标/队列/几何，不是完整模型采用。
M12没有reuse，本地项未作用，其未解释差值保留。配对中位数不等于两个n0增量中位数之差。
所有原始/冻结预测/evaluation结果保存width16_local_holdout，远端两场均已结束。

随后新增full_local_transfer/replay.py，在已有完整224expert anchor上做三层回顾消融，
无全计划拟合：legacy共享响应，增加已有8T小行密度响应，再给所有16T任务W2后续段
加local0.28。独立成本、全局gather系数、旧首请求/零后续发出需求、历史31入口场景不变。
legacy阶段端点与原完整模型差<1e-7us，三组完整回放/请求守恒正常结束，Ruff通过。
| 模型 | 完成预测/us | 两场误差/% |
|---|---:|---:|
| legacy | 27477.503 | -4.252/-4.066 |
| family8 | 27475.980 | -4.257/-4.072 |
| family8+local16 | 27482.750 | -4.234/-4.048 |

三者均预测lane0最后完成，与该anchor实测一致，但绝对时间仍明显低估。
本地项只使这条完整计划的预测增加6.770us，不能把前瞻小实验的收益外推为整体问题已经解决。
results.json保留所有lane/31场景事件，identity.json保留模型源。下一主项仍是全局gather
以及长GEMM后续需求：当前背景后续段发出请求为零，与已测大M持续流量不符。

只读硬件核对：lscpu -C及sysfs确认L2为1280KiB，L3为71680KiB，两个当前域共享CPU分别
240..279与280..319；保存full_local_transfer/cache_geometry.json。
8T W13 owner1MiB已占L2容量80%，16T owner512KiB占40%，还未计A/output等；
不能仅凭B大小推断全部后续访问命中私有缓存，更不能据此设后续共享请求为零。
这只是容量与共享范围事实，不是miss率或带宽定律。首次通用只读查询自动审批超时未执行，
一次简化重试及明确sysfs读取成功，无硬件或运行配置修改。

本轮无新模型参数、Production/default或远端基准变化；完整预测与planner搜索验收仍未完成。


## L2与DDRC分离观测：16T联合计数器烟测通过，正式采集启动

为约束后续段的私有缓存之外请求，先只读核对Arm内核PMU别名：
armv8_pmuv3_0/type=8，format/event=config:0-15，l2d_cache=0x0016、
l2d_cache_refill=0x0017、l2d_cache_wb=0x0018。没有猜测PERF_TYPE_RAW或事件编码。
新增独立l2_stage_demand16，复用large16_stage_demand/native_v2，SHA仍
96e9772c570bbd9470252b552d1e2ec886733dda73ed00acbdd6b07e0c111fdd；没有重编译或修改kernel。
在原64个DDRC事件之外增加CPU304..319每核上述3事件，共112事件。
核心事件逐个以其enabled时间归一化；保留每核和聚合事件率/每调用事件数，不直接换算精确LLC字节。
它们是worker CPU上的system-wide计数，包含barrier/运行时/系统活动，不作为纯B访问归因。

Python计数器校验通过合成归一化、缺事件拒绝、running/enabled不足拒绝检查，Ruff与shell
语法通过。真实13格单轮112事件smoke_pmu正常结束（exec52078 exit0），analyze.py检查
全grid、数值/route guard、CPU、二进制、全部core/DDRC coverage及running/enabled>=.99通过，
B32活动格L2 refill非零。原始烟测已同时保留本地和远端。

随后执行`bash tmp/joint_cost_model_20260911/l2_stage_demand16/run.sh`，exec32669当前活跃：
无PMU控制612361，再PMU612361/612362，两场串行，M12/48/1718、W13/W2、B1/B32及idle，
每场5warmup+31round，共13×36×3格。M1718窗口500ms，其余100ms，64call lead-in，
完成边沿计数及一调用量化比例继续保留。单轮smoke不用于流量/性能结论。

16T/NUMA3/CPU304..319/controller240、H4096/F512/BF16/SVE256/Ntile16、
(16,0,0,1,1)、stage8/4MiB、owner512/256KiB与原独立探针一致。
常数输入/权重与synthetic连续route仍不同于真实4-copy/scrub路径；只补整阶段请求特征，
不宣称已测得panel时序、LLC命中率或资源服务容量，不更新T0/模型参数/默认planner。


## 16T L2/DDRC首场完成：两层请求随M的增长不同

原exec32669继续运行，control及session1均已complete并收回；analyze.py control/session1
通过完整grid、数值/CPU、二进制、112事件coverage与running/enabled检查。
第一场计时相对控制最大绝对差0.5765%，尚需第二场重复性检查，不替换真实T0。

B32每调用L2 refill事件，M12→M48：W13 129349→530225（约4.10倍），
W2 62485→205943（约3.30倍）；相应DRAM read为6.3098→12.7938MiB（约2.03倍），
2.6492→5.5842MiB（约2.11倍）。同一变化下两个观测量增长不同，不能仅用一个DRAM压力
代理所有本地请求。shape_ratios_session1.json保留精确比值，没有把事件数换算成LLC字节。

M1718每调用事件（L2 refill/writeback）：W13 B1约3870764/37721、B32约10887463/147811；
W2 B1约809176/408132、B32约1407033/377032。相应DRAM读/写MiB：
W13 B1 24.8887/-0.0198、B32 48.7426/0.2205；W2 B1 2.6347/1.0891、B32 11.4964/2.7812。
单份B的大M仍有大量refill活动，但system-wide事件也含同步/运行时活动，不唯一归因到A/B/output。
这些是整阶段观测，不能直接给每个panel分配请求或认定某个硬件miss机制。
近零负idle-subtracted写值继续保留，不作为负物理需求。

本地准备l2_stage_demand8：复用已有large8_stage_demand/native（80c46a...），
M12/48/1205、CPU312..319，24个core事件+64DDRC共88事件；500ms大M/100ms其余、
B1/B32及控制/两PMU协议保持，拟seeds612381/612382。几何为(8,0,0,1,1)，
stage8/4MiB、owner1MiB/512KiB，常数A/B/连续route和既有buffer容量不变。
Python Ruff和run.sh语法通过，目前仅本地准备，未同步、构建或运行；等待16T第二场终止。
16T第二场仍在原进程中采集，没有并行基准、参数更新或Production/default变化。


## 16T L2/DDRC两场完成；8T 88事件正式采集启动

exec32669返回session2 complete/COMPLETE/exit0，16T控制和两PMU共1404格完成。
第二场收回后analyze.py session2通过112事件、grid、数值/CPU、binary和窗口检查。
第二场相对控制最大计时偏差0.9952%（M12/W2/B1，绝对约0.50us），不将其唯一归因于PMU。
B32 M12→M48的refill增长约4.19倍（W13）、3.33倍（W2），DRAM读增长约2.04/2.03倍，
与第一场两层请求增长不同的结论重复；仍不能从总数推断每panel时间分布。

M1718两场每调用refill事件：W13/B1 3870764→4222818（+9.10%），
W13/B32 10887463→10486566（-3.68%）；W2/B1 809176→806320（-0.35%），
W2/B32 1407033→1537233（+9.25%）。W2/B1 writeback408132→410007（+0.46%），
同期DRAM读量变化约-11.97%。因此L2与DRAM观测不能合成同一个固定倍率，也不是所有L2点都稳定。
M48/B1 W13 refill11799→41733，W2 refill4316→6848，writeback186→2282；
这些较小基数上的变化也保留，不能只展示稳定的大M条件。
repeat.json保存逐条件对比，不把事件数作为精确LLC字节或纯B请求。

确认16T终止后同步l2_stage_demand8并执行13格单轮88事件烟测，复用既有8T native80c46a...，
无kernel改动或重编译。analyze.py smoke_pmu通过事件coverage/running、数值/CPU及窗口检查，
原始烟测已保留本地和远端。随后启动run.sh，exec86703当前活跃：control612381后
PMU612381/612382，两场串行，5warmup+31round。CPU312..319、NUMA3、M12/48/1205、
W13/W2、B1/B32、stage8/4MiB、owner1MiB/512KiB、(8,0,0,1,1)、Ntile16、
500ms大M/100ms其余窗口与已有8T探针一致；没有并行基准或模型参数/默认planner更新。


## 8T L2/DDRC首场完成；观测请求库保留单位与适用边界

exec86703输出session1 complete，第二场继续原串行脚本。控制与第一场已收回，
analyze.py control/session1通过完整grid、binary、数值/CPU、88事件及窗口有效性检查。
第一场PMU相对控制最大计时差0.2318%，M1205最少W13/W2完成32/65次，计数比例继续保留。
8T/B32每调用refill：M12 W13/W2 132701/64667，M48 509533/238176，
M1205 10781516/1593861；M1205/B1 W13/W2仍2993991/511409。
相应M1205/B32 DRAM读/写MiB为W13 40.7937/0.5874、W2 15.0617/6.8966；
B1为12.2740/-0.0176及3.0972/3.7408。事件与字节保留各自单位，不作精确LLC字节换算。
本场不足以宣称8T跨会话稳定性，等待第二场。

新增request_budget_bank/build.py，为已完成数据建立供后续拟合读取的观测库。
目前width16.json完成：12个(M,stage,copies)格，每格保留两次measurement_instance的
DRAM读/写MiB和L2 refill/writeback事件向量、service、计数比例和PMU/control差；
不把不同场次各资源独立平均，也不把8T/16T的同session编号当作同一次配对测量。
保留负idle估计和state漂移，temporal_segments为null、real_path_transfer_validated为false。
lookup仅返回观测描述，未知M或要求尚未测量的段内分布会明确报错，不静默填零。
12格/两向量、支持点查询、未知形状/时间分布拒绝检查及Ruff通过。
命令`.venv/bin/python tmp/joint_cost_model_20260911/request_budget_bank/build.py 16`。
8T库需等第二场完整分析后生成；没有将该观测库直接替换运行时cost或修改默认planner。


## 8T L2/DDRC重复性闭环：请求库完成，状态桥接仍是接入条件

2026-09-12续查远端session2.stdout到round31/31，pgrep未发现该实验measure.py/run.sh进程；
结合原exec86703已返回COMPLETE/exit0，确认正式采集终止，无重启或新基准。
rsync收回session2全部文件后执行：
` .venv/bin/python tmp/joint_cost_model_20260911/l2_stage_demand8/analyze.py session2 `；
` .venv/bin/python tmp/joint_cost_model_20260911/request_budget_bank/build.py 8 `。
前者通过原grid、数值/CPU、88事件coverage、binary及窗口有效性检查；后者通过12格×两场
完整性和未知M/段内分布拒绝检查。全部control+两PMU共1404格已完成。
协议沿用上节8T/NUMA3/CPU312..319/controller240、M12/48/1205、W13/W2、B1/B32、
H4096/F512/BF16/SVE256/Ntile16、(8,0,0,1,1)、stage8/4MiB、owner1MiB/512KiB，
相同native80c46a...，5warmup/31round，100ms/500ms窗口，无kernel修改。

两PMU场相对既有control的最大计时偏差为0.2318%。B32六格L2 refill第二场相对第一场
变化为+0.18/+0.20/+1.99/+0.31/-0.81/-0.27%（依次M12 W13/W2、M48 W13/W2、
M1205 W13/W2）；DRAM读量六格变化绝对值均不超过1.66%。
M1205/B32 W13 DRAM读40.7937→40.3440MiB/call，W2读15.0617→15.3116、
写6.8966→7.7708MiB/call（写+12.68%）。M1205/B1 W2读3.0972→2.2695、
写3.7408→2.8008MiB/call，refill511409→453790（-11.27%）。
M48/B1 W2 writeback996→3417事件/call；小基数与状态漂移都保留，不能以稳定B32读量
概括所有资源。近零写量的百分比无独立实际意义。两场不构成总体置信区间。

request_budget_bank/width8.json与既有width16.json均保存整阶段观测向量、单位、来源、
独立measurement_instance及适用边界；width8_repeat.json保存两场逐格原值与变化。
没有把独立探针的常数输入/连续route/B1或B32映射为真实4-copy/scrub路径，也没有将L2
事件换成精确LLC字节。temporal_segments仍null，real_path_transfer_validated仍false。

决策：可以继续使用分离的本地请求代理与DRAM读写特征；不把表直接接入完整模型。
下一项实验应桥接实际W13→W2状态并约束长阶段请求分布，优先覆盖当前关键路径的大M及
8T/16T实际owner条带；同时保持真实无竞争T0冻结。若独立/真实路径需求不一致，先解释
状态差异，不通过提高竞争敏感度抵消。随后重点处理global gather响应，避免继续围绕只
改变整计划约6.8us的local系数微调。完整模型误差、未见计划和planner搜索验收仍未完成。
本轮仅收回并分析已完成数据、生成观测库及记录结果，无Production/default或模型参数变更。


## 真实路径请求采样窗口审计：先做大M，不用SIGSTOP改变阶段历史

2026-09-12只读导航核对：common/fused_moe_bf16_tiled.cpp中现有SIGSTOP位于独立GEMM
微基准的setup/整段循环边界，不是实际expert W13→W2的阶段计数接口。
ddr_phase_barrier_native.cpp是合成读取探针，也不能作为真实GEMM路径。此次不修改这些入口。

对width16_anchor_missing/session1/2_compact.json的12个isolated形状、两场各31pair，
验证isolation_validated/bitwise_correctness及16worker数，然后计算同阶段所有worker区间
交集[max(start),min(end)]，空交集记0。结果保存real_path_request_bridge/window_audit.json。
这是现有真实路径的窗口可行性审计，不是新PMU测量或请求量估计。
16T M1718的交集中位数W13为10746.40/10751.82us，W2为5221.45/5230.03us，
gather为345.23/349.63us；M12 W13只有82.16/81.40us，W2为41.95/41.68us。
M12和M77的gather交集中位数为0，不能假定存在全worker同时处于gather的纯区间。

下一实现约束据此收敛：先用持续外部PMU采样与真实trace时间对齐检查大M阶段内部，
不在W13/W2之间暂停进程或添加采样屏障。正式采集前测量读取计数器所需时间及跨事件
读取跨度，核对trace与采样时钟；只有完整采样边界落入目标阶段交集的记录才作为该阶段
内部观测，跨边界记录保留为混合区间。另核对其他task是否正在执行，不能只凭目标worker
交集宣称全NUMA DDR计数纯属目标。DDRC尾部排空与其他系统流量仍需控制/扰动检查。
阶段内部采样不等于整阶段请求预算，不将缺失的首尾区间外推为零，也不据此称已测panel
请求。小M/尾块需要另一种已验证的测量粒度；本轮不先验承诺能用相同采样周期解析它们。

本轮为只读诊断与D类结果记录，L0引用/格式检查；未改模型参数、native、默认planner，
未运行新远端基准。该审计消除了直接复用独立SIGSTOP入口的错误路线，下一步是计数器
读取开销和时钟可行性烟测，再决定是否进入大M真实路径采集。


## 外部PMU live-read开销实测；现有trace缺少绝对原点

2026-09-12新增Lab脚本real_path_request_bridge/read_overhead.py，复用冻结
l2_stage_demand16的specs与PerfCounterSet；仅读取已开启计数器，不在每次读取时disable/reset。
用已有私有fd列表进行24-byte读取，核验长度、count/enabled/running单调、相邻读取
running/enabled>=0.99；context manager释放资源。不修改共享helper/native/model。
Ruff通过；远端确认相关基准无运行进程后，执行
`numactl --physcpubind=240 --membind=3 .venv/bin/python tmp/joint_cost_model_20260911/real_path_request_bridge/read_overhead.py`。
Arm-codex-internal既定root/venv，core事件CPU304..319，NUMA3 16DDRC，50warmup+1000reads。
两组均exit0，原始1000次跨度保留read_overhead_session1.json：
32 DDR读写事件median106.735us/P99 111.970us/max136.750us；
112完整事件median506.805us/P99 527.520us/max533.630us。
这是空载机器上顺序读取整组的跨度，包含Python/syscall及计数器访问；不是工作负载扰动验收，
也不是所有事件的共同瞬时采样。大M可继续研究，小M W2约42us无法用当前整组读取解析。

源码审计profile_utils.h使用std::chrono::steady_clock；MoeTraceCollector保存origin_，
write_report将phase begin/end减去origin_输出start_ms/end_ms，但MOE_CALL不输出绝对origin。
因此尚不能把Python monotonic_ns直接与trace相对时间对齐；时钟种类相似不等于已建立映射。
下一实现需独立Lab构建提供绝对trace原点并验证平台时钟一致性/新增记录扰动，保持冻结默认
扩展不动。采样记录必须保留读取开始/结束界限；只有跨两次读取的整个保守区间均位于目标
阶段且无其他任务活动时，才讨论对应资源流量。此前连续采样方案仍未完成真实路径验证。

本轮E类采样诊断，影响范围仅新脚本与结果；资源回收、计数完整性及目标机运行通过，
没有kernel数值变化或模型改进声明，没有Production/default切换。回滚边界为该独立Lab目录。


## 绝对trace原点Lab对象已准备并开始编译

2026-09-12 E类诊断扩展：本地与Arm common源码SHA同为c87a9bd0...，默认扩展仍dd554ea3...。
real_path_request_bridge/fused_moe_bf16_tiled_origin.cpp只在MOE_CALL格式增加
origin_steady_ns，并输出已有origin_.time_since_epoch()的纳秒数。源改动仅涉及报告格式/参数，
不增加kernel执行期间的clock调用，不改Production源码。origin_source_identity.json保存完整SHA。
现有queues_runner/analyze_workspace_isolated_width.py按键解析MOE_CALL header，源码检查
允许新增字段；完整运行兼容性尚未验证。compile_origin.py复用现有build.ninja编译参数，
保持最后-O2/-march=armv8.6-a+bf16+i8mm，额外include原common目录解决相对头文件路径。
原对象与默认so均不覆盖，输出在Lab目录；已有目标文件会拒绝重编译覆盖。
Ruff通过。远端numactl CPU240/NUMA3启动单对象编译，exec23563已轮询仍活跃；
日志compile.stdout/compile.stderr与精确命令compile_command.txt保留。尚未链接或运行候选。
下一步链接独立扩展、验证绝对时钟映射/数值/trace解析和扰动。没有新的cost-model性能结论。

随后exec23563返回exit0，Lab common对象编译完成。尚未链接；先前“编译活跃”状态由本终态替代。


## Lab原点扩展链接与65调用真实路径烟测完成

2026-09-12 link_origin.py读取原build.ninja中的compile对象并追加原MoE native对象，
仅替换common对象；依赖与默认so的readelf NEEDED及setup.py相符。link_identity.json保留
命令和全部对象SHA。候选独立so SHA为6e5eb6a267de5a28faebb6d6303976644a1c8d66be356f9199148631c364f726，
链接exec23162 exit0，独立importlib加载通过；默认so未覆盖。对象复用的源码/构建等价性和
默认候选直接输出对照仍需后续验证，加载成功不代替这些门槛。

run_origin.py在导入包前显式加载候选fused_cpp._moe_C，封装Python调用前后monotonic_ns，
调用未修改queues_runner。frontier.json及baseline.json仅将extension_sha改为候选；原route、
计划、workspace与协议不变。原baseline本地缺失，已从远端读取保存original_baseline.json；
首次启动又因缺PYTHONPATH在kernel之前退出（exec43813 exit1），保留smoke.stderr。
随后显式PYTHONPATH=src:.用smoke_v2全新输出重试，exec26508 exit0。
命令为Arm NUMA3/CPU240..319、OMP/MKL/OPENBLAS1，run_origin.py measure加
--frontier real_path_request_bridge/frontier.json、--experiment-baseline同目录baseline.json、
原request016_case017 route、--workspace-max-tokens2048 --max-plans13 --nested-routes
--correctness-only --phase-trace smoke_v2.trace --output smoke_v2.json。
13计划×(一参考+四权重副本)=65调用，实际H4096/F512、BF16/SVE256、原full-stripe计划，
固定workspace poison与reuse检查全部bitwise通过；这不是新旧so逐位对照或计时重复实验。

validate_origin.py验证全部65call连续性、420550个phase数量/顺序边界；每个origin_steady_ns
以及由origin+相对时间恢复的phase起止都落入对应Python monotonic_ns调用区间，全部通过。
origin_validation.json和smoke_call_bounds.json保存结果；大trace保留远端同目录。
这支持当前平台时钟口径的粗边界一致性，不宣称纳秒级同步精度或PMU扰动已合格。
Ruff通过，新脚本均独立Lab文件；Manifest更新实际状态，无模型参数或默认变更。
下一步做冻结默认与候选的直接正确性/扰动对照，再使用连续PMU区间采集大M真实阶段。


## 原点扩展直接输出/扰动对照已冻结并启动

2026-09-12新增real_path_request_bridge/paired，E类诊断验证，无模型参数改动。
默认dd554ea3...与候选6e5eb6a2...分进程加载；frontier仅保留原anchor、iso_m0012、iso_m1718，
两arm baseline/frontier除了扩展身份一致，实际route、nested转换、plan bridge、权重seed20260906、
输入seed314159保持。run_arm.py对15个正确性前缀调用输出完整SHA；指纹只在正式预热前计算，
各调用monotonic边界保留。冻结design.json：5warmup+31pair，seeds612401/612402，
顺序default1/candidate1/candidate2/default2，每arm123调用。计时比较仍可能含跨进程状态波动，
因此报告两场及MAD，不将差值唯一归因于新增字段。
门槛为每场每计划compute-end中位数差<=2%，每目标阶段差<=max(2us,2%默认)。
全部15个前缀输出SHA须跨arm相同，内部poison/reuse逐位检查也必须通过。门槛在采前固定。
score.py从完整trace取全任务最后W2端点，避免compact仅保留部分expert造成漏计；分别评分
expert96 gather/W13/W2，核验全部候选origin/phase处于对应Python调用区间。外层Python时间
含trace格式化/输出，不用来替代计算时间。此对照不验证PMU采样自身扰动。
Ruff、run.sh语法通过。执行`bash tmp/joint_cost_model_20260911/real_path_request_bridge/paired/run.sh`，
Arm-codex-internal NUMA3/CPU240..319，H4096/F512/BF16/SVE256/Ntile16，既定实际full stripes，
4权重副本/216MiB scrub/固定workspace/early merge off，OMP/MKL/OPENBLAS1，PYTHONPATH=src:.。
exec41078轮询输出default1 complete，首arm123call分析通过，candidate1正在原串行脚本内运行。
没有并行基准、重启或默认so替换；评分必须等待四arm全部终止，当前不宣称扰动门槛通过。


首场candidate1随后完成，score.read_arm在远端解析两个完整trace，核验123调用、候选绝对
时间边界及15个新旧前缀输出SHA全部相同。interim_session1.json保存初步结果。
anchor compute-end默认28923.63us、候选30071.37us（+3.968%）超过2%门槛；
isoM12 gather64.08→68.48us（+4.40us）也超过2us门槛。isoM1718 W13
10813.91→10816.54us（+.0243%），W2 5227.30→5229.66us（+.0451%）均通过。
这不是整体扰动通过；等待原exec41078继续candidate2/default2反向顺序，保留首场失败。
不能以大M目标阶段相近替代完整anchor门槛，也不能尚未定位就归因于输出字段。


## 原点扩展ABBA完成：扰动门槛未通过，首场anchor增时在间隔而非GEMM

exec41078最终default2 complete/COMPLETE/exit0。四arm各123调用全部原分析器验证；
远端score.py完成（exec89255 exit0），两场各15个前缀新旧输出SHA一致，所有候选origin/phase
仍在Python monotonic边界内。evaluation.json总门槛false，不能采用为已验证的正式PMU载体。
完整anchor compute-end第一场28923.63→30071.37us（+3.968%，MAD35.72/51.44us），
第二场28986.07→28907.28us（-0.272%，MAD46.75/21.62us）。首场差异未在反向顺序重复。
M12 gather第一场64.08→68.48us（+4.40us），第二场73.61→58.37us（-15.24us），均失败；
M1718 gather第二场501.03→488.41us（-12.62us），超过冻结10.0206us界限。其他评分项通过。
保留所有失败，不凭首场M1718 GEMM相近或第二场anchor通过改变既定门槛。

新增lanes.py/gaps.py完整trace诊断，按anchor224expert的所有阶段聚合，不用只覆盖少数任务
的compact推断完整lane。两场各31anchor调用，检查3×224阶段与调用序列，Ruff通过。
第一场lane0候选减默认均值：完成+1127.735us、初始入口+31.870us，gather-9.335us、
W13-217.785us、W2-117.714us，阶段之外间隔+1440.698us；各均值分项可加。
间隔分解为任务前+829.559us、gather后+382.551us、W13后+228.589us。多个后期任务各有
约20–27us任务前增量，不是单一expert GEMM变慢。第二场lane0间隔仅+16.038us，
完成均值-78.168us。lane_diagnostic1/2.json和gap_diagnostic1/2.json保留逐lane/任务结果。
这是时间账单定位，不能唯一归因于频率、调度、barrier或新增输出字段；没有系统计数证据。

所有小JSON/log收回本地paired，原大trace仍在Arm同路径；正式采集及后处理均终止。
当前decision：候选加载、内部/跨arm前缀正确性与时钟边界通过；完整扰动验收失败。
下一步用同二进制独立进程A/A刻画入口/gather/间隔场次变动，并在相同状态控制下重复A/B，
不以平均两场抹平失败或将门槛改成宽松值。PMU真实路径正式采集仍不启动；模型参数、
无竞争T0、默认扩展与planner均未改变。完整cost-model与planner目标仍未完成。


## gather跨场变化拆解：M12主要是worker到达，不是访存服务

2026-09-12先复用已完成ABBA做同二进制跨场比较，same_binary_repeat.json保留结果。
默认两场isoM12 gather64.08→73.61us（+9.53us），isoM1718 484.63→501.03us（+16.40us），
均超过原阶段门槛；候选M12 68.48→58.37us（-10.11us）。两场seed不同、期间运行过另一arm，
所以这只是回顾性同二进制比较，不冒充新随机A/A因果对照，也不撤销原AB门槛失败。

完整worker_details拆解保存gather_components.json（每格31样本、16worker）：
M12按default1/candidate1/candidate2/default2顺序，包络64.08/68.48/58.37/73.61us，
到达跨度60.50/65.40/55.81/70.89us，平均worker服务2.866/2.934/2.873/2.879us。
小M包络的大幅变化主要跟随到达跨度；不能把整段包络波动用于重拟合访存T0或响应。
M1718同顺序平均worker服务415.536/416.486/416.629/423.149us，
到达跨度62.56/66.59/65.34/70.22us，默认第二场同时有服务与进入变化，尚非单一原因。
上述量都是各自中位数，不把它们当作严格可加分解。动态入口层与执行服务层继续分开。

为控制seed，采前冻结paired/aa_design.json：默认同一dd554ea3二进制，default3/default4
相邻独立进程，同seed612403、相同三计划、5warmup+31pair，原几何/输入/权重/workspace协议。
run_aa.sh仅改变运行cell与seed，复用原run_arm和trace分析器；bash语法通过。
执行`bash tmp/joint_cost_model_20260911/real_path_request_bridge/paired/run_aa.sh`，exec17814当前
活跃，两场串行。预期246调用，输出均新文件。后续同时报告原包络门槛及worker服务/进入
诊断，不按新结果放宽原门槛。无参数拟合或PMU正式采样，无默认扩展修改。


## 同seed默认A/A完成：复现约1.12ms anchor增时与16T间隔状态

exec17814返回default4 complete/COMPLETE/exit0，两场各123调用原分析器通过。
score_aa.py（Ruff通过）复用全trace compute-end提取，验证seed均612403、二进制SHA相同、
plan/pair/copy顺序相同、15个正确性前缀输出SHA相同；exec31492 exit0。
aa_evaluation.json总门槛仍false：anchor28947.55→30071.23us（+1123.68us，约3.882%，
MAD60.79/88.23us），M12 gather61.17→66.46us（+5.29us）超过原门槛。
M12平均worker gather2.83125→2.81125us（-0.020us），到达跨度58.36→61.32us。
M1718整计划44554.63→44372.16us（-182.47us）仍在原2%门槛内；其他原评分项通过。
这是相邻同seed默认进程也会出现类似增时的直接证据，不将AB失败改判通过。

复用lanes.extract/gaps.extract对default3/default4解析，exec57399 exit0；
aa_lane_gap.json保留完整分解并收回本地。lane0均值完成+1035.347us，初始入口-2.728us，
gather+26.559us，W13-199.965us，W2-234.716us，间隔+1446.197us。
间隔分解任务前+773.488us、gather后+396.935us、W13后+275.774us，与此前AB首场
+1440.698us间隔模式接近。均值账单与整体中位数分开，不作错误相加。
因此下一诊断对象是默认运行时可变的同步/任务进入状态，不能把约1ms增量拟合到DRAM响应。

只读源码核对：ThreadBarrier是generation自旋屏障，4096次后std::this_thread::yield；
ResidentThreadPool::run的调用线程在fn(0)之前显式绑定tid0的core，之后才等待其他worker。
因此“额外未绑定controller同时占用worker核”不是这段运行路径的直接解释。
这不证明具体barrier/yield或cache-line机制；下一实验需可区分的同步诊断证据，保留kernel
与独立T0不动，不通过改变模型参数吸收该运行状态。所有本轮采集/分析已终止，PMU正式
采样仍未启动。默认扩展、模型和planner未改；完整目标未完成。


## 16T逐worker切换诊断：共同释放之前的延迟，仍需区分记录与barrier

2026-09-12新增paired/worker_gaps.py，读取默认A/A两场完整trace，覆盖每场31anchor×42个
lane0 expert，每阶段检查16个CPU240..255完整/无重复。exec1361 exit0、Ruff通过。
每种转换1302样本，保留default3/default4_worker_gaps.json；不将同一call内expert视作
独立实验重复来估计置信区间。
gather→W13：最早下一阶段开始减最晚上一阶段结束的均值1.491→10.942us，
下一阶段worker start跨度均值0.363→0.467us；W13→W2前者0.836→7.402us，
后者0.534→0.538us。最晚启动CPU分散，没有单一固定CPU独占该现象。
因此主要增量发生在共同下一阶段开始之前，而非下一阶段已有worker开始后某个worker
迟迟加入。结论限于trace边界；不排除上一阶段结束时间戳之后有worker迟到barrier。

源码核对实际fused-silu路径：gather trace_phase_end后执行barrier.wait，再采W13 begin；
W13 trace_phase_end后执行barrier.wait，再采W2 begin。record_phase先采end时间，再执行
buffer lookup/emplace_back、记录字段与CPU元数据。因此“阶段外间隔”同时包含trace记录
后半段、barrier与调度，不可直接命名为纯barrier成本。CPU affinity元数据每buffer首次初始化，
已固定CPU时后续record_cpu直接返回缓存值；不是每阶段调用pthread_getaffinity的简单解释。
下一可辨识对照应把trace-record开销与barrier释放分开，保持GEMM及其他控制相同；
未测出机制前不添加任务数惩罚、不修改独立T0或竞争系数。
本轮仅离线解析已有数据及只读代码，未启动新基准；结果记录与脚本检查通过。


## trace记录完成边界诊断已构建；正确性烟测启动

2026-09-12应用performance-ablation技能及variant-contract，选择可分离阶段计时，未删除
barrier/atomics或GEMM指令。新增E类trace_record_timing独立Lab目录和manifest项。
父版本为原点候选源码，新增MoePhaseTraceRecord.record_done_time，并在record_phase原字段
写入后采一次profile::now；PHASE追加record_done_ms。原阶段end/begin定义不变，保留全部
原访存与同步工作。design.json保存源码身份、有限实验计划和停止条件：正确性烟测后最多
两场三计划诊断；若慢状态不出现或干预改变运行状态，结果为不充分，不继续追求完全分解。
新增clock与record布局变化是明确干预混杂，不能把结果直接迁移回默认模型或称固定指令槽消融。

复用此前单对象编译/链接脚本，只把独立路径改为trace_record_timing。exec78540
compile+link exit0；原编译参数和未改对象身份保留，默认so没有覆盖。Python Ruff通过。
prepare_cases.py从既定三计划default frontier/baseline复制并只改扩展SHA；不改变任务/输入。
run_origin.py复用先前显式模块加载/调用边界封装；validate_origin.py要求15调用并额外检查
每个record_done时间满足phase_end<=record_done<=Python调用结束。旧parser兼容性需运行验证。
在Arm NUMA3/CPU240..319、H4096/F512/BF16/SVE256/Ntile16、原full stripes、4权重副本、
固定workspace、early merge off条件启动--correctness-only烟测，exec77631当前活跃。
尚无该诊断的正确性/扰动/时间归因结论，不修改T0、竞争响应或planner。下一步先等待原烟测
终态和全部时间顺序校验，再决定两场诊断采集；不将新时间戳视为无扰动测量。


## 记录边界诊断两场完成：未复现慢状态，按预设条件停止

exec77631烟测exit0：15call/97050phase内部逐位正确性、origin/phase/record_done时序检查通过。
诊断so SHA299b43a11f1b179f95ccea35b506b370a0b1b068223c447f337a5d7795069e89。
随后run_session.py在正确性前缀保存输出SHA，正式测量前完成指纹计算。run.sh两场同seed612403，
三计划anchor/isoM12/isoM1718、5warmup+31pair，原Arm NUMA3/80workers/H4096/F512/
BF16/SVE256/Ntile16/actual full stripes/4copies/216MiB scrub/workspace/early merge off协议。
exec62413最终session2 complete/COMPLETE/exit0，两场各123call原trace解析与隔离检查通过。
analyze_record.py两场exit0，15个前缀SHA均与默认default3一致，全部phase满足
Python_begin<=origin+phase_start<=origin+phase_end<=origin+record_done<=Python_end。
每场31anchor×42lane0 expert×2转换=2604行，均验证同一转换的end/done/next边界可相加；
不将这些行当作独立场次。Python Ruff和shell语法通过。

均值（us，session1/session2）：
| 转换 | 总间隔 | 最晚end→最晚record_done | 最晚record_done→最早next |
|---|---:|---:|---:|
| gather→W13 | .58965/.52753 | .29959/.31083 | .29005/.21671 |
| W13→W2 | .55051/.49098 | .26253/.25346 | .28798/.23752 |

两场均未复现默认慢状态中10.942/7.402us转换间隔，甚至比默认快场1.491/.836us更短。
新增clock与record布局/编译变化造成的干预混杂不能排除，不能据此宣布默认慢状态的记录成本
只有.3us或barrier成本只有.2us。按design.json预设最多两场与状态变化停止条件，终止此项
归因实验，保留为不充分结果，不追加样本追逐慢状态，不将这些时间纳入T0/竞争拟合。
剩余区间仍含函数返回、下一clock、同步和调度，不是纯barrier延迟。

首场后处理短暂在第二场进程准备期间运行，不能宣称全过程无并行CPU活动。文件时间核对：
session1_record_analysis完成04:45:24.748 UTC，session2.trace创建04:45:29.996 UTC，
后处理在第二场首个trace输出前约5.25s结束；仍保留准备期干扰可能，不据此强化无扰动结论。
第二场分析在采集终止后执行。全部小JSON/log收回本地，大trace保留远端同目录。
下一路线应回到资源需求/竞争模型主问题，将尚未识别的进入与同步状态作为明确不确定性，
不再把这一项诊断当作可以无限扩展的前置条件。模型/默认扩展/planner未修改，完整目标未完成。


## 回到gather竞争响应：worker服务标定改善回顾误差，三队列仍低估

2026-09-12新增gather_response_worker/audit.py，对已有width16_response（长背景）、
width16_wave_response（小/长/小队列）及width16_local_holdout（两队列）两场各31pair，
检查CPU240..255的16worker并分别计算服务均值、包络、到达跨度的配对增量/MAD。
M1718混合队列worker服务增量：n1为46.30/48.85us，local n2为104.74/104.73us，
local n3为181.94/185.40us，cross n3为178.20/175.13us；local n3到达变化仅-.74/-1.36us。
仅长GEMM背景local n3服务增量为-8.35/-8.15us，保留负控制，不当作零噪声或真实加速规律。
这组强竞争减速不同于前述进入时间变化，足够支持回到执行服务响应层。

M类Lab参数对照：gather_response_worker/simulate.py复制当前local_domain engine，仅追加
每worker gather start/end观测，不改事件步进、请求、资源反馈；在训练计划上剥离新增字段
后与原engine完整返回值精确相同。fit.py冻结原T0、历史入口、所有legacy first-Q/zero-reuse
请求及其他系数，只拟合gemm_to_gather。design.json采前冻结训练为已看过的wave/session1
M1718/local1，区间[0,.016]、12次二分，以平均worker服务配对增量为目标；不按包络拟合。
系数由原.0002到.00176953125，训练预测46.338us，对实测46.2975us。训练点贴合不作泛化结论。

运行audit.py/fit.py/evaluate.py完成，Ruff通过。排除唯一训练记录的其余51条既有记录：
竞争增量MAE22.426→6.883us，绝对worker服务MAE21.302→8.074us。
按数据集增量MAE：长背景2.613→2.613us；两队列22.935→3.706us；混合队列
42.926→14.395us。三队列M1718仍预测约121.7us，对实测175–185us；local n2预测86.57us，
对实测104.74/104.73us。保留强竞争低估，不宣称一个线性系数解决全部响应。
所有评估数据此前已被检查，这是回顾性留出条件，不是新鲜holdout。候选保留用于下一结构
对照；legacy后续零需求缺陷未解决，系数也不能解释为硬件常量或测得需求模型。

本轮没有新远端采集、生产或默认planner修改；新参数尚未完整计划回放。下一步检查按请求
压力的非线性响应是否能解释n2/n3，同时保留长背景与其他M对照，随后必须完整计划迁移。
不要把队列数直接用作运行时需求特征，也不将未识别同步状态吸收到这个系数。


## 请求率二次gather响应：局部增量更准，完整anchor低估缩小但仍未解决

2026-09-12 M类Lab新增gather_response_quadratic，采前design冻结训练为既有wave/session1
M1718/local1与local3，固定独立成本、请求与其余参数；参数非负，a∈[0,.016]、b∈[0,.4]，
12次外层/12次内层有界二分。gather速度分母增加b*(q_K/100)^2，q_K为其他expert当前实际
GEMM发出率KiB/us，包含原减速反馈；100仅是数值尺度，不是硬件容量，不用任务数。
拟合a=.001220703125、b=.083203125；n1/n3预测46.2716/181.9333us，对训练46.2975/181.9356us。
两训练plan的b=0与原worker-observer engine完整输出相等；独立常速发出方解析gather结束
19us检查、独立无干扰10us、负/NaN参数拒绝通过。Ruff通过，原请求守恒/收敛拒绝保留。

其余50已见记录（同一排除集比较）：线性→二次竞争增量MAE5.8166→2.5827us，
绝对worker服务MAE7.4989→8.5958us。两队列集合增量3.7056→3.7776us，绝对3.4578→9.3798us；
混合队列增量11.5321→1.3540us，绝对8.8211→6.3271us；长背景2.6127/9.9157us基本不变。
local n2 M1718预测109.530us，对104.742/104.727us；cross n2预测109.562us，对98.574/85.996us。
全部训练/非训练记录和残差保留evaluation.json；不是新鲜holdout，不宣称所有分组改善。

绝对误差的账单：冻结M1718 worker gather均值416.25625us（model.json来源
width16_anchor_missing/CPU240..255），wave两场n0实测392.987/392.446us，n2数据控制
395.330/393.733us。独立基线比这些带M77前缀的控制高约21–24us。原竞争低估抵消了部分
基线高估；二次补足增量后抵消减少，绝对误差可变大。原因尚未唯一定位到前缀状态/缓存等，
不能直接修改T0或回调竞争系数来维持抵消。中位数配对差与两个中位数之差仍区别处理。

随后replay.py完成完整224task anchor、31个冻结入口场景的原/线性/二次三臂回放，
exec24356 exit0。8T小行敏感度与16T local W2 .28均固定，无完整计划拟合；原臂全部事件
端点与此前full_local_transfer精确到1e-7us，新增worker观测字段除外。
| 响应 | 完成预测/us | 对原两场实测误差/% |
|---|---:|---:|
| 原响应 | 27482.750 | -4.234/-4.048 |
| worker线性 | 27665.478 | -3.597/-3.410 |
| worker二次 | 27803.196 | -3.117/-2.929 |
三臂均lane0最后完成。M1718 gather包络预测544.853→695.262→829.542us，
worker平均服务475.095→613.895→739.184us；W13/W2预测基本不变。原实测gather约958us，
所以仍有未解释缺口，整体也仍低估约839–895us。results.json保留全部事件与lane。

结论：二次请求率响应作为有条件Lab候选保留，已完成回顾全anchor迁移；未通过新计划、
新数据或搜索验收。请求表达仍是首窗口预算/后续零请求，16T标定向8T接收方迁移未独立验证。
下一步优先检验带前缀的gather基线状态并补充真正未见的压力组合，避免用基线误差抵消
竞争误差；不得据整体误差下降切换默认。无新远端采集、Production、planner默认变更。


## gather前缀状态同场对照准备并启动

2026-09-12 E类Lab新增gather_prefix_baseline，先核对既有source：独立M1718与prefix-n0
均为expert96/16T/CPU240..255/full stripes，区别包含前置expert230/M77与采集场次。
不直接把跨场21–24us差值写进T0。prepare.py复用原width16_response路由变体和anchor，
对M12/48/1718各生成direct、prefix_local、prefix_cross，加anchor共10plans。
direct使用既有isolate_bridge并归约依赖；local复用原n0；cross只将prefix16T放到CPU280..295，
target仍240..255、target依赖prefix，其他任务在target之后。保留全部expert/route工作，
CPU共享祖先与bridge身份检查通过；无kernel/默认扩展变更。Ruff与shell语法通过。

采前design.json冻结两场seeds612421/612422、每场5warmup+31pair、10计划随机配对，
主要观测为worker平均gather执行的同pair差与MAD，辅助阶段包络/到达/W13/W2及实际不重叠。
跨域prefix同时改变放置和时机，不作为唯一私有缓存归因；效应若不重复则保留状态不确定性。
不据一个实验自动重拟合独立成本或竞争响应。
执行`PYTHONPATH=src:. bash tmp/joint_cost_model_20260911/gather_prefix_baseline/run.sh`，
exec91240当前活跃；默认dd554ea3扩展、原route request016_case017/layer4、H4096/F512/
BF16/SVE256/Ntile16、NUMA3/80workers、(16,0,0,1,1)前台/前缀几何、stage8/4MiB、
owner512/256KiB、4copies/216MiB scrub/固定workspace/early merge off保持。
每场预期410调用，两场串行；正确性前缀通过后才进入正式测量，数值失败会终止脚本。

analyze.py已准备并Ruff通过，要求410调用原隔离验证、seed/frontier身份、每格31pair，
检查所有prefix在target前完成及实际CPUlocal/cross匹配。分别输出direct→local、direct→cross、
local→cross的worker服务/包络/进入差；不将中位数误作可加账单。目前没有新测量结论。


首场随后session1 validated，410调用数值/隔离检查通过。收回compact后本地analyze.py1
通过前缀时间顺序、实际CPU与31pair检查。M1718 direct/local/cross worker服务中位数
411.840/389.591/412.727us；local-direct配对差-21.344us、MAD1.974us，cross-direct
+0.869us、MAD1.763us，cross-local+23.311us、MAD2.272us。
M12 local-direct+.344us、cross-direct+.720us；M48分别+1.417/+2.966us。
首场支持带本地前缀的大M状态差异，但小M方向相反，不使用统一前缀时间奖励。
跨域控制使“只有等待同样前缀就变快”的解释不足，但尚不能唯一确认私有缓存机制。
exec91240第二场继续，尚未更新独立基线或竞争参数。analysis_session1.json保留全部指标。


## gather前缀两场完成；复用scratch与输入交集审计、同LLC异team对照

exec91240最终session2 validated/COMPLETE/exit0，两场各410call通过数值与原trace隔离验证。
第二场收回compact后analyze.py2通过31pair/前缀不重叠/CPU检查。
M1718 local-direct配对worker差-21.886us（MAD5.551），cross-direct-4.728us（MAD8.138），
cross-local+16.744us（MAD6.549）；第一场对应-21.344/+0.869/+23.311us。
两场支持同team本地前缀的大M减时；跨域差值较小但第二场噪声增加，保留全部值。
M12第二场local/cross相对direct+.289/+.748us，M48为+1.100/+2.589us，仍与大M方向不同。
同场配对重复效应不等于跨场绝对T0相同：第二场M1718 direct中位数420.596us，第一场411.840us。

读取原route并验证SHA，复用nested_prefixes审计输入token集合（input_overlap.json）：
M12/48与prefix230的77个token交集为0；M1718交集只有7个（.407%，逻辑input57344bytes）。
这是地址集合事实，不是实际缓存命中/流量/节省时间，不能直接宣称全部收益来自相同输入。
源码ensure_scratch_config按(core_begin,threads)合并scratch；ResidentScheduledScratchPool
使用相同key复用单元；gather写scratch.packed_a。故现有local/cross同时改变LLC、owner CPU
和scratch身份，不能仅标记成LLC历史效应；scratch/output历史是待区分解释。

采前冻结gather_prefix_neighbor/design.json，保留原10控制并增加M12/48/1718三个
prefix_neighbor：前缀16T core_begin16即CPU256..271，与target240..255同LLC但不同team/scratch。
复用原nested路由、数据和后继依赖，validate_resources通过；全部13plans，bridge SHA重算。
同LLC异team仍联合改变私有缓存ownership和scratch，不能单独区分二者。
seeds612441/612442、5warmup+31pair、原几何/默认dd554ea3扩展/4copies/scrub/workspace协议。
run.sh语法通过，source frontier与设计保存独立目录；前一采集终止后执行
`PYTHONPATH=src:. bash tmp/joint_cost_model_20260911/gather_prefix_neighbor/run.sh`，exec27726活跃，
两场预期各533call串行。暂无新测量结论，不修改T0或竞争参数；目标是给状态特征找到适用范围。


## 同LLC异team首场：M1718收益未出现；完整anchor的历史作用范围

2026-09-12为gather_prefix_neighbor准备analyze.py，扩展原验证至533call/13plans，
加入prefix_neighbor CPU256..271及direct/local/neighbor/cross六组配对关系；Ruff通过。
exec27726输出session1 validated，收回compact后本地analyze.py1通过所有实际CPU、前缀
先后关系与31pair检查。M1718 local-direct配对-20.694us（MAD5.324），neighbor-direct
-.5225us（MAD3.926），cross-direct-2.171us（MAD4.178）；neighbor-local+20.8075us
（MAD3.8475）。单纯同LLC异team没有重复同team的大M收益，保留第二场验证。
M12 neighbor-direct+.563us，M48+1.9625us；不能用固定20us奖励覆盖所有M。
这些对照区分了“同LLC就足够”的解释，但仍未区分私有owner状态与scratch地址复用。
第二场在同一exec27726中串行继续，暂不修改基线或竞争参数。

同时只读审计原完整224task jobs：按(core_begin,width)计算上一共享scratch任务，验证其
确实是依赖祖先；共9个无本调用前驱的scratch根。full_anchor_history.json保留逐任务描述，
不是物理缓存驻留或温度模型。关键expert96/M1718/core_begin0/16T是该scratch的首任务，
没有本调用前缀。因此不能把前缀约-21us修正直接施加给完整anchor中expert96，拿它解释
该任务仍低估的gather；前缀状态当前主要影响竞争标定的绝对基线及后续任务。
此审计限制修正的作用范围，避免以整体残差反推无依据的历史奖励。未做新的模型拟合。


## 同LLC异team对照完成；新队列顺序响应验证已冻结并启动

exec27726最终session2 validated/COMPLETE/exit0，两场各533call通过。第二场compact收回后
analyze.py2通过实际CPU/前缀时间/31pair检查。M1718 local-direct配对-23.153us（MAD5.027），
neighbor-direct+1.283us（MAD5.516），cross-direct+.785us（MAD3.594），neighbor-local
+23.727us（MAD6.465）。与首场同team约-20.7us、异team接近0的方向一致。
保留同team/scratch历史为候选状态，未分离私有owner与scratch复用，不拟合任意M公式。

回到竞争泛化：prepare_gather_response_holdout.py复用原63个背景expert和三条8T队列，
前台M96/1718、固定M77前缀。三种顺序保持每条队列成员不变，长任务位置分别
balanced10/10/10、frontloaded0/0/0、staggered0/10/20；每M另有none控制，加anchor共9plans。
重新构造拓扑和依赖，nested背景/前缀实际计数、CPU资源、expert集合/full stripes校验通过。
这是未测frontloaded/staggered顺序，balanced为既有结构参考；不按新数据调整候选。

freeze.py在采集前完成8条件×3模型×31历史入口场景（exec70580 exit0），冻结原/线性/二次
系数、T0、legacy请求、运行时常数以及所有预测与来源SHA。没有前缀基线修正、没有实测起点、
不使用新n0重新预测。M1718二次worker服务none416.256、balanced597.696、frontloaded416.256、
staggered525.089us，预测相对none约181.44/0/108.83us。
M96二次服务none22.0969、balanced31.7311、frontloaded22.0969、staggered27.8669us。
同任务数/组合下响应随实际动态请求重叠变化；frontloaded零增量也是对legacy零后续需求的
可证伪预测，不能保证其成立。

frozen_predictions.json采前门槛：frontloaded/staggered新条件每场按M分别比较竞争增量，
M96 MAE<=5us，M1718<=15us，二次不得恶化同集合线性MAE；同时保留绝对服务与W13/W2误差。
这是局部响应验证，不代替完整计划/search门槛。Ruff和run.sh语法通过。
前项基准终止且预测冻结后执行
`PYTHONPATH=src:. bash tmp/joint_cost_model_20260911/gather_response_holdout/run.sh`，exec81092
当前活跃：seeds612461/612462、5warmup/31pair、每场369call两场串行。默认dd554ea3扩展、
原route/H4096/F512/BF16/SVE256/Ntile16、NUMA3/80workers、目标16T owner512/256KiB、
背景8T owner1MiB/512KiB、stage8/4MiB、4copies/216MiB scrub/workspace/early merge off保持。
没有新模型参数或默认planner变更；本次预测未见新的硬件数据。


## 新顺序首场冻结评分：二次未胜线性，保留frontloaded负增量

2026-09-12新增gather_response_holdout/evaluate.py，Ruff通过。读取冻结预测并逐项验证
source_hashes，要求369call/seed612461/31pair/CPU240..255与原数值隔离通过。
第一次评分在rsync(exec21899)仍传输时触发FileNotFoundError，没有生成evaluation；
等待原传输exit0后重跑评分exit0，未重启采集、修改模型或覆盖原始数据。
exec81092已输出session1 validated，第二场继续原串行脚本。

首场新顺序(frontloaded/staggered)增量MAE（原/线性/二次，us）：
M96为1.2091/1.0904/1.6678；M1718为47.3704/9.1294/14.9808。
二次均低于绝对5/15us阈值，但劣于同集合线性，故两形状冻结综合门槛均false。
M1718 balanced实测184.666us（MAD9.339），二次181.440；staggered实测91.831us（MAD4.356），
线性约86.532、二次108.833；frontloaded实测-12.960us（MAD3.295），三模型预测近0。
M96 balanced/staggered/frontloaded实测5.953/2.684/.249us，对二次9.634/5.770/0。
不能用已有balanced条件贴合替代新顺序验证。保留frontloaded负增量，不称作负压力或把
竞争系数拟合成负数；历史状态变化与基线不确定性仍需单独解释。
二次M1718绝对worker服务误差balanced+18.744、staggered+39.263、frontloaded+34.853us，
不能只报告竞争增量的较小数值。全部阶段误差和冻结门槛见evaluation_session1.json。
尚无第二场结论，不调整参数，暂不采用二次响应为后续默认；原完整模型/搜索目标未完成。


## 新顺序两场完成：二次响应不推广，线性保留为对照

exec81092最终session2 validated/COMPLETE/exit0，两场各369call数值/隔离通过。
等待第二场rsync exec13957明确exit0后，本地evaluate.py2通过全部冻结来源SHA与31pair校验。
第二场新顺序增量MAE（原/线性/二次，us）：M96 1.1066/1.1785/1.7559，
M1718 45.1379/6.8969/14.3877。二次仍未胜线性，两个形状门槛均false，与首场相同。
M1718 balanced实测183.753（MAD8.319），staggered90.192（MAD4.836），frontloaded
-10.134（MAD2.083）us；冻结二次181.440/108.833/0，线性staggered86.532us。
M96实测balanced5.265、staggered2.493、frontloaded.235us，冻结二次9.634/5.770/0。
第二场二次M1718绝对worker服务误差balanced+25.024、staggered+44.812、frontloaded+36.746us。
W13/W2绝对误差在evaluation中单独保留，未用新控制点调T0或更新预测。

decision.json明确quadratic_not_promoted：旧队列/旧完整anchor回顾改善不代替新顺序泛化，
两场均未满足冻结比较门槛，故不推广二次为通用gather响应。不是所有gather修正都被否定：
线性相对原模型在新M1718顺序上明显减小增量误差，但没有通过完整planner采用验收。
保留线性作后续对照；下一诊断优先需求时序和前台M/阶段敏感度，不继续加曲率拟合本holdout。
frontloaded负增量两场均出现，不能作为负竞争压力处理；历史/基线状态与非负响应需分层。
全部本轮采集与传输/评分已结束。默认planner、独立T0与模型参数未更改。


## 真实时序条件诊断：只修阶段重叠不能挽救二次响应

2026-09-12新增E类gather_timing_diagnostic/analyze.py，本地读取新顺序两场已完成trace，
未重新拟合参数。前台各worker start使用实测；背景gather每worker沿实际区间分配既有逻辑
请求预算；背景GEMM保留既有首段Q并用整阶段实测/独立时长比例推算首段窗口。
小M只有一段，窗口等于实际阶段；长M是均匀伸缩假设，不是实测panel时序，后续请求仍为零。
因此该诊断同时给定时序与假设的pacing，不是纯起点干预，也不是实际PMU请求曲线。
按各边界积分固定线性/二次响应，保持T0不变；记录预测worker结束越过真实结束的数量，
不得将真实未来背景与反事实前台同时当成自主反馈模拟。无新远端采集。

常压解析脉冲19us、无竞争10us及等价脉冲拆分不变检查通过；初始Ruff格式问题已修正，
Ruff通过。两场分析exec56742/63052 exit0，session1/2.json保留全结果。
新顺序四条件（两个M合并）的MAE，自主→条件（us）：线性第一场5.110→4.596，
第二场4.038→4.578；二次第一场8.324→11.760，第二场8.072→11.801。
M1718 staggered二次从自主108.833us变为条件121.382/122.716us，对实测91.831/90.192us，
高估变大。线性条件94.424/95.445us，比实测高2.593/5.253us。
M1718 balanced线性条件128.128/128.971us仍比实测184.666/183.753低约55–57us；
二次条件191.526/194.603us则高约7–11us。frontloaded两模型仍预测零增量，未解释实测
-12.960/-10.134us的历史/基线相关差值。所有已见控制也保留，未只取有利结果。

结论限于当前legacy需求假设：只改善阶段起止/整体pacing不足以解释二次泛化失败。
请求量、阶段内分布与前台敏感度仍是未分离因素；不能断言真实时序毫无影响或硬件机制
已经定位。二次不推广决定不变，线性保留对照。下一步应约束需求/接收响应表达，不继续
对同一验证集调曲率或把真实trace引入正式预测。完整cost-model/search目标仍未完成。


## 阶段组成与前台划分审计：相近名义请求率不等于相同减速

2026-09-12新增gather_timing_diagnostic/exposure.py，沿前项真实worker区间/均匀伸缩首窗口
假设，将名义暴露分为W13、W2、gather；另记录长GEMM后续段重叠team-us。Ruff通过，
exec43407两场本地分析完成，输出exposure.json。请求KiB仍是模型预算，不能称PMU实测。
前台实际持续时间用于统计，只作描述，不作为正式预测输入或回归特征泄漏。

balanced条件M96 worker窗口主要只有W13首请求，W2暴露中位数0；M1718同时有W13/W2，
W13占比约.663/.690。逐pair按暴露总量/worker平均时长计算后取中位数：
M96名义GEMM率170.226/168.031KiB/us、配对服务增量/n0为28.40/24.89%；
M1718名义率173.535/174.897KiB/us、相对增量46.87/47.08%。
receiver_comparison.json保留数值与口径。相近名义率下减速不同，但前台M、所有权划分与
背景阶段组成同时变化，尚不能唯一归因为某一个敏感度。

joint_gather_work.py的实际工作分配条件为panel数>=width采用完整panel，否则采用K划分。
16T M96为8个完整panel，M1718为143完整块加尾块，二者确实处于不同分配区间；
这提供比任意M惩罚更明确的候选分类，但没有新拟合或默认行为变化。
frontloaded条件后续W13重叠约为3倍前台时间，staggered约为1倍；legacy此段零发出，
名义首请求暴露在frontloaded为0。重叠team-us不等于流量，已测整阶段持续请求也不能直接
按比例填入这些区间。不能凭负减速把后续请求认定为物理零。

下一有辨识力的对照应固定前台划分，分别施加W13/W2背景并比较实际请求率；再固定背景
跨前台panel/K划分检查敏感度。现有数据的两因素共变不足以支持继续增加自由系数。
本轮为已有数据的E类离线特征诊断，未启动远端基准或改动模型参数；二次拒绝决定保持。


## W13/W2背景与16T gather划分边界：受控探针构建、烟测启动

2026-09-12 E类Lab新增gather_stage_control16，复用gather_io_states实际common gather快照
和匹配实现TU，不改gather算法；前台driver从8worker改16，CPU304..319，route改为原
layer4 expert96的1718个真实flattened route位置，输入H4096/token2048、32input/32output槽。
保留padding校验中的物理8/12stride，只改变worker循环、barrier参与数和记录数组尺寸。
前台M96/180/181/192/1718：M180与M181跨panel数>=16的实际所有权边界。
background复用已有JIT stage driver，支持W13/W2选择；仅移动CPUbase到304-8*n，
允许1/2/3team（本grid为1/3），与前台互斥且全部在LLC280..319。控制进程CPU240。
核函数源不改，M12背景stage8/4MiB，8T owner1MiB/512KiB、Ntile16/full stripes。

32格grid：idle1、五形状无背景5、五形状×两背景数量×两stage20、四背景solo4、
M1718三队列B1控制2；默认B32。该探针是稳态独立IO状态，不继承真实W13→W2或前缀状态，
不能替换真实T0。每个前台worker耗时中位数与team调用时间分别记录，不混作真实单调用包络。
保留64call lead-in、100ms观察、DONE后持续执行到ACK、数值/guard/CPU检查。

build.sh foreground实现沿原common匹配O2/no-SVE-autovec参数、driver O2/SVE256；背景沿
既有O2/JIT构建。exec46076 exit0，独立foreground/background完成；binary_identity.json
保留实际二进制SHA，默认扩展/原探针均未覆盖。Python Ruff与shell语法通过。
validate_smoke.py要求32格完整唯一、16worker正样本、数值/CPU标志和64event计数运行率>=.99。
前台hash按新binary_identity验证；无生产环境变量或全局build变更。

随后在Arm-codex-internal既定root/venv、NUMA3/CPU240..319启动两轮串行烟测：
measure.py --foreground本目录foreground --background本目录background --seed612480
--rounds1 --warmup0，先--no-pmu smoke.jsonl并validate_smoke，成功后smoke_pmu.jsonl与验证。
exec84471当前活跃。尚无烟测结论或正式性能数据；通过执行和counter检查后才确定正式采集。
下一决策为固定前台比较W13/W2及实测请求率、固定背景比较前台所有权边界，不增加自由系数。

随后exec84471返回exit0：无PMU及64event两轮各32格完整烟测均通过binary/grid、数值/CPU、16worker记录与计数器运行率检查。原始JSONL及日志收回本地。仅验证测量有效性，尚未开始正式重复采集。


## 16T前台×W13/W2背景正式重复采集启动

2026-09-12在两轮32格smoke全部通过后，冻结formal_protocol.json与现有二进制：
foreground ed9fd64ab8c8cfc2ac225c247c667de316e5ee490359d1e76c99b3d4f5a1f365，
background 0f38384414a0f54d8b9911be4eadecbc6ee6de4fe716811bce959dcb72979a4d。
执行`bash tmp/joint_cost_model_20260911/gather_stage_control16/run.sh`，exec71732活跃。
control612481无PMU，然后session1/2 seeds612481/612482带64事件，全部串行，
每场32格×(5warmup+31round)，100ms窗口，共3456格。输出新文件，已有文件拒绝覆盖。
前台M96/180/181/192/1718、16T CPU304..319，背景M12/W13或W2、1/3×8T CPU304-8n..303，
NUMA3/controller240、同LLC280..319，输入/输出32槽、背景B32及M1718 B1控制，
原实际gather和JIT算法/条带保持。只采集独立响应，不替换真实T0或改变模型参数。

新增analyze.py，沿既有gather_with_gemm分析逻辑扩展stage维度、16worker与新binary身份。
在已完成smoke/smoke_pmu上完整grid/数值/CPU/计数器检查通过，Ruff通过。
保留每事件own enabled归一化和idle-subtracted读写、背景solo每调用流量、前台配对服务差/MAD、
背景吞吐反馈和每窗口一调用量化比例。联合PMU流量不直接归因前台；B1没有对应solo点，
其背景rate-change明确null而非0。空载扣除的负估计保留，不当负物理需求。
所有烟测分析只验证分析器，不作为正式性能结论。正式控制场仍在原进程中运行，尚未评分。


## 16T阶段背景控制场完成；完整配对分析准备

2026-09-12扩展analyze.py保留worker中位数的max/min以及team median，避免M181不均匀
tail仅由worker均值表述；这不是单次team envelope或可加中位数。正式输入要求31round/5warmup，
二进制身份固定。两个既有smoke完整load验证max>=mean>=min>0、team时间>0通过，Ruff通过。
新增compare.py准备W13/W2同M/n配对响应、背景solo实测read比率、联合读写、PMU/control
计时差；明确相同n不等于相同流量，不通过联合减solo归因前台。ownership_features.json保留
五形状逐worker逻辑工作量，不称实际off-core字节。没有修改运行中的采集器或二进制。

exec71732输出control complete后，收回control.jsonl，等待rsync exec27617 exit0再分析。
analyze.py control通过全部32×36=1152格数值/CPU/grid/二进制与样本完整性。正式31round
worker平均服务（us）：无背景M96/180/181/192/1718为15.051/38.165/37.143/39.562/408.434。
三个W13 B32背景的配对增量分别2.997/15.657/12.587/13.379/128.124us，
M1718 MAD .970us；三个W2对应2.629/13.506/10.477/11.487/103.126us，M1718 MAD1.075us。
一个W13/W2背景M1718增量29.512/25.709us；背景吞吐相对solo分别-2.630/-2.489%，
三个背景分别-3.315/-3.705%。控制场没有PMU，不能把W13/W2减速差解释成相同请求率下
的阶段敏感度差。M180→181响应变化同时含划分/尾块与不均匀工作，不能只归因于一个因素。

当前exec71732继续session1/session2两场PMU原串行脚本，尚未完成。control_analysis.json
保存所有32格原样本/摘要，不用这组独立稳态时间替换真实路径T0。下一步等待实际流量与
重复性后再比较，不拟合新系数。默认模型/planner/扩展保持。


## 控制场边界诊断：平均值之外，慢worker的响应也变化

2026-09-12继续轮询既有exec71732，PMU session1仍活跃（日志已到round15/31），
未重启或并行基准。新增gather_stage_control16/boundary.py，复用完整control.load验证，
按同round比较M180/181的worker mean/max/min和team median，Ruff通过。
control_boundary.json保存全部无背景、1/3team、W13/W2配对统计。

无背景M180/181的max(worker median)为40.57/40.31us，mean为38.165/37.143us，
min为36.69/7.77us；M181小尾worker使均值变化，不能只看平均数。
三个W13背景max为58.40/54.17us，对各自n0的配对相对响应43.87/33.80%；
三个W2为56.17/52.04us，相对38.32/28.33%。最慢worker的变化也不同，故不只是
小尾worker拉低平均值。M181 min约7.77us且竞争变化接近0；其工作量与其他worker不同。
team median含本探针pthread同步，W13 n3为70.47/68.12us，不当作真实运行时gather包络。
这些都是control31round的独立稳态观察，未测等流量条件，不能唯一归因划分策略或缓存。

后续资源响应需要保留worker工作量/所有权/服务差异，以预测max完成而非只调整均值。
本轮未拟合模型，正式PMU数据尚未完成，默认planner不变。


## 16T W13/W2控制第一场PMU完成：实际流量不同，前台自身负载也不同

exec71732输出session1 complete，第二场继续。收回session1.jsonl并等rsync46763 exit0后，
本地analyze.py session1、compare.py session1、boundary.py session1全部通过完整1152格
identity/grid/数值/CPU/64event与运行率检查。27个前台格的PMU/control配对计时相对差中位数
最大绝对值.7627%（M180 n0），不唯一归因为PMU开销；第二场重复尚未完成。

背景solo idle-subtracted read：1team W13/W2为48.805/42.286GB/s，3team为136.162/115.410GB/s；
3team配对W2/W13比率.8460（MAD.0042），不是等请求率条件。
M1718前台1team W13/W2配对服务增量30.405/25.697us，3team132.121/104.239us；
M96三team3.024/2.589us。不能从同n的差异直接推出W13/W2具有不同接收系数。
联合计数包含前后台：M1718三team W13 read/write176.691/23.653GB/s，
W2为158.875/24.787GB/s；不将joint-solo解释成前台实际流量。

前台单独idle-subtracted read/write（GB/s）：M96 5.636/2.047，M18025.439/14.677，
M18125.243/13.837，M19226.364/14.375，M171856.678/30.367。
单独occupancy/command proxy分别约32.735/50.222/50.917/51.801/60.798；它不是CPU访问延迟。
M1718三team B1背景W13/W2服务增量-.012/-.363us，整体读写与无背景近似；保留这些近零
负配对值，不称真实加速或负压力。大幅减速主要出现在B32控制，而非仅有后台线程。

这一组观测要求后续检查“联合资源状态相对前台自身无竞争资源状态”的变化，而非仅将
所有自有流量从响应特征中减去。这里是待检验结构方向，不是新公式/参数或硬件容量结论；
不同前台服务方式仍可能有不同敏感度。第一场不足以采用，所有统计保留session1分析/比较文件。
第二场仍在既有exec71732串行采集，无模型或默认planner改变。


## 16T阶段背景正式采集闭环：两场流量/响应重复，队列代理不直接等于接收延迟

exec71732最终session2 complete/COMPLETE/exit0，control+两PMU共3456格正式采集完成。
等待第二场rsync64732 exit0后，本地analyze.py/compare.py/boundary.py session2全部通过。
两场均完整1152格、固定二进制/shape/CPU/64event与运行率校验。第二场27前台条件
PMU/control配对相对计时差中位数最大绝对值1.853%，第一场.763%；不称精确PMU因果开销。

第二场背景solo三team W13/W2读135.323/114.258GB/s，与第一场136.162/115.410重复，
相同team数仍不是相同流量。第二场三team W13/W2前台配对服务增量(us)：
M96 2.831/2.431，M18015.459/12.897，M18112.016/10.141，M19213.438/11.539，
M1718129.217/100.518；第一场M1718132.121/104.239。保留边界/形状差异，未按这些点重拟合。
完整逐pair、计时/PMU对照和max/min/team统计均保存本地session1/2_analysis/comparison/boundary。

新增queue_diagnostic.py（Ruff与完整输入验证通过），描述同形状n0到竞争条件的聚合
occupancy/command变化及相对worker减速。第二场M180：W13 n1 proxy+3.337、减速11.59%，
n3 proxy+37.444、减速39.97%；W2 n1 proxy+1.931、减速10.46%，n3 proxy+28.063、减速33.29%。
简单相除不能得到统一稳定的线性敏感度。聚合proxy混合了前后台/控制器的请求权重，
基线到联合运行的组成也变化；它不是前台访问延迟，小差值分母尤其不能用于精确系数。
queue_diagnostic.json保留全部40条件描述；未建立新硬件队列定律或拟合模型。

决策：该独立控制数据支持以实际流量与worker工作量区分输入/阶段，但未完成等流量W13/W2
因果比较，也不把稳态探针直接迁移为真实路径T0。下一模型对照应保留前台自身读/写负载及
非均匀worker需求，先做无竞争极限/请求守恒，再检验泛化；不能只用队列数或聚合proxy替代。
本轮全部采集/传输/分析已结束，无远端活跃基准，无模型参数、默认扩展或planner更改。


## 自身无竞争负载归一化对照：oracle小幅改善，solo预测尚未优于简单对照

2026-09-12新增M类Lab gather_self_pressure，design.json在拟合前声明回顾训练为
stage_control16/session1、M96/192/1718的W13 B32 n1/n3六格，其余M180/181、W2与session2
为既有非训练条件；B1单列，缺BG solo不参与共同比较。所有数据已见，不称新鲜holdout。
结构h(P)=1+aP+bP²，P=(read+w*write)/100GBs，响应h(Pjoint)/h(P0)-1。
100仅数值归一化；a/b非负网格0..0.4步.01，w∈{0,.25,.5,1,2}。外部read-only对照
使用a*q_bg+b*q_bg²，相同a/b网格，参数数和输入信息不同，不作为严格同复杂度比较。
fit_oracle.py从实测joint/solo流量和实测T0输入，故联合流量已含实际减速反馈，是oracle结构诊断。
训练得到外部a=.11/b=.08；self-normalized a=.01/b=.10/w=2（w在预设边界，不能解释为
硬件写代价，也未扩大搜索范围）。34共同非训练格增量MAE2.3163→1.7680us。
相同压力/基线返回零变化及正负载方向检查通过，Ruff通过；无硬件容量或自主模型结论。

predict_solo.py进一步冻结上述参数，只输入各角色单独测量的流量/耗时，求前台自身反馈：
cycle(v)=max_worker_T0/v+overhead0，overhead0=solo_cycle-max_worker_T0；
前台read/write速率按solo_cycle/cycle(v)缩放，预算每调用保持。背景仍固定solo流量。
以二分求解v=h(P0)/h(Ptotal)，无背景极限v=1检查通过。近零负BG idle写估计仅在物理输入
截到0，原值保留样本；未用联合计数输入预测。一个共同速度缩放所有前台worker仍是限制，
不是完整tail/工作量响应或任意plan模拟。
34非训练格增量MAE2.7386us，总read/write误差MAE2.4239/1.2030GB/s，不优于外部对照2.3163us。
M1718非训练格平均有符号误差：减速+8.0578us、总读+4.6280GB/s；M180减速约-3.0545us。
solo_prediction.json保留全部条件，未只报告oracle改善。背景已测吞吐下降几个百分点而
该原型将其固定为solo值，因此反馈缺项值得单独检验，但尚非唯一归因。

下一步先加入背景速率反馈再检验总流量和服务是否同时改善，不回调T0抵消流量偏差。
当前不推广该结构，不把实测joint压力接到正式planner；无新远端采集或默认修改。


## 双角色反馈对照与每调用流量账单：速度反馈不足以解释全部联合流量

2026-09-12新增gather_self_pressure/coupled.py，在既有h参数a=.01/b=.10/w=2与solo/T0
固定下，前台vF=min(1,h(PF0)/h(P))，背景vB=1/[1+gamma*max(0,h(P)/h(PB0)-1)]；
前台cycle=max_worker_T0/vF+overhead0，背景group读写按vB缩放。总负载P由两角色速率自洽求解，
标量二分60次、残差<1e-9检查。BG solo基线仍来自同n的组测量，不是任意任务组合的独立构造。
采前coupled_design.json限定gamma∈[0,1]、24次二分，只用原六格训练集内的session1
M1718/n1/W13背景吞吐变化-2.6346%标定；未新增前台拟合点或改变原非训练排除集。
gamma=.121234566。gamma0与旧solver三种输入精确到1e-9、无背景vF=1、无overhead对称角色、
负/NaN gamma拒绝检查通过，Ruff通过。

同34非训练格：前台增量MAE2.7386→2.3122us，总read MAE2.4239→2.6625GB/s，
write MAE1.2030→1.1684GB/s，背景吞吐变化MAE.3513百分点。仅时间有所改善，总read未同步
改善，且外部简化对照MAE2.3163us几乎相同。不能据此宣布新结构更好或迁移到完整planner。
统一前台速度、固定每调用需求、已测BG group baseline等限制均保留在coupled_prediction.json。

进一步budget_audit.py不使用上述响应方程：用每角色实测joint调用率乘该角色solo每调用
read/write预算，再与实际joint流量比较。40个B32竞争格完整读取/校验，Ruff通过。
M180/n3/W13两场actual-minus-reconstructed read6.860/5.643GB/s、write2.260/1.827GB/s；
W2对应read7.135/6.574、write2.364/1.868。M96/n3/W13 read3.439/2.696、write2.445/2.364；
M1718/n3/W13 read1.415/.589、write.083/.086GB/s。budget_audit.json保留逐pair/MAD和
一调用量化比例，不把小差值都视为机制信号。

两个角色的实际吞吐已用于重构，残差不能再仅归于模型漏了背景减速。它提示solo每调用
下层请求预算可能随竞争/缓存状态变化，或仍有窗口/估计误差；并不唯一归因前台、RFO或
某级cache。Native与PMU窗口不完全同起止，full-contained调用数有边界量化，仍作为限制。
下一诊断应检验请求量的状态依赖，而非继续只调整服务减速。该判断也不授权将所有joint-solo
差直接填入前台需求。无新远端采集或默认模型/planner变更；当前结构仍不推广。


## 输入/输出复用状态×竞争需求实验：烟测通过并启动正式采集

2026-09-12 E类Lab新增gather_demand_states，只改Python命令网格，不改任何native。
复用gather_stage_control16的foreground ed9fd64...、background0f383844...，CPU/NUMA/算法不变。
M96/180前台input copies1/32×output copies1/32，各配none或3team M12/W13/W2 B32；
M1718保留IO1/1、32/32的none/W13 n3控制，另idle与W13/W2 solo，共31格。
输入/输出活跃地址集合和复用距离同时变化，不能解释成唯一缓存层消融或实际路径T0。
每个native原本支持这些copy参数，继续检查padding/guards、16worker与CPU位置；grid包含
所有必要的按IO状态n0参考，背景solo为同stage/n3。design.json在采前冻结。

无PMU与64event两轮各31格smoke串行执行，exec28095 exit0；validate_smoke均通过grid、
二进制、数值/CPU、worker与计数器运行率检查。原始JSONL/log收回本地后，新六维analyze.py
在两smoke上完整通过（输出smoke_analysis/smoke_pmu_analysis），Ruff通过。
新的cell key显式包含input/output copies，参考使用本状态n0，避免将不同活跃工作集混作
同一个独立成本。烟测仅确认测量有效性，不用于流量/性能结论。

随后formal_protocol.json冻结control612501、PMU612501/612502，每场31格×(5warmup+31round)，
100ms，共3348格；原二进制不变，已有文件拒绝覆盖。run.sh语法通过，执行
`bash tmp/joint_cost_model_20260911/gather_demand_states/run.sh`，exec46303当前活跃。
Arm-codex-internal既定root/venv，前台16T304..319，背景3×8T280..303，同LLC，NUMA3，
controller240，H4096/BF16/SVE256、前台真实expert96 route；背景stage8/4MiB、owner1MiB/512KiB。
control后两PMU全部串行。没有新模型拟合或默认变更。
主要分析为双方实测调用率缩放各自solo预算后的总流量残差、input/output交互及服务变化，
保留窗口量化/共同counter归因限制。正式数据尚未完成，不预先把残差指定给foreground。


## IO状态控制场完成：竞争减速具有正交互，等待流量闭环

2026-09-12新增gather_demand_states/interactions.py，复用六维完整load，以同round计算
Δ(32,32)-Δ(32,1)-Δ(1,32)+Δ(1,1)。Δ为每状态相对自身n0服务增量；PMU场还将双方实际
调用率用于solo预算重构，再计算总read/write残差及其IO交互。残差按FG调用率归一化只是
单位选择，不作FG流量归因。两轮完整smoke数据上16state/stage cells与4交互解析通过，Ruff通过。

active_footprints.json按实际gather_work记逻辑输入/输出地址量：M96 IO1/1=1.5MiB，
1/32或32/1=24.75MiB，32/32=48MiB；M180对应2.8125/46.40625/90MiB；
M1718 1/1=26.890625MiB、32/32=860.5MiB。3team B32背景活跃B为W13 768MiB、W2 384MiB。
这些不是分配总大小、有效cache容量或实际DRAM需求，不能由容量直接推断命中。

exec46303输出control complete；等rsync14336 exit0后本地analyze.py control与
interactions.py control通过完整31×36=1116格、binary/shape/CPU与31正式round检查。
M180 W13 n3下，IO1/1、1/32、32/1、32/32的服务配对增量分别-.1806、2.2456、
1.3744、15.5213us；同round交互中位数11.720us、MAD.4525us。W2交互10.140us、MAD.5313us。
M96 W13对应-.05875/.265/.4644/2.7856us，交互2.0463us、MAD.1275us；W2交互1.8906us。
不把单项中位数直接相加代替配对交互；也不把近零负值称作真实优化收益。
输入/输出同时轮换的竞争变化具有明显非加性，但控制场尚无PMU，不能直接断言流量放大
或唯一cache机制。原控制样本与摘要/交互完整保留。

当前exec46303继续两场PMU，未重启或并行采集；尚不修改需求量函数或响应系数。
下一步以PMU残差核验该服务交互是否对应状态相关请求变化，不能只拟合额外时间项。


## IO状态第一场PMU完成：额外写流量跟随输出轮换，服务/流量交互不等价

2026-09-12先补control_completion_interactions.json，确认交互不限于平均worker：
M180 W13/W2的max(worker median)服务增量交互9.51/8.76us，team median交互10.59/9.36us；
M96分别2.66/2.47及2.30/1.98us。team时间包含探针同步，不当作真实path包络。

exec46303输出session1 complete，等待rsync47054 exit0后本地analyze.py session1和
interactions.py session1均通过完整1116格、31正式round、binary/CPU/64event检查。
28个前台格相对control的worker时间配对相对差中位数最大绝对1.2247%（M96/W2/32-32），
保存session1_pmu_control.json，不唯一归因于PMU开销。

第一场M180 W13 n3，IO1/1、1/32、32/1、32/32的服务增量分别-.2163/2.1625/1.4675/15.0675us；
双方实测joint调用率缩放solo预算后，read残差-2.414/1.917/-.599/5.453GB/s，
write残差.019/2.771/.016/1.937GB/s。额外写主要随输出轮换，输入单独轮换没有同样写残差。
M96相应write残差.0146/2.2564/.0206/2.1639GB/s，同方向；不将这些聚合残差唯一归因FG。
1/1的负read残差也完整保留，提醒预算/窗口/背景流量均可能变化，不强制截零或填入需求。

同round交互：M180 W13服务11.3169us（MAD.4313），read残差交互2.301GB/s（MAD1.636），
write残差交互-.841GB/s（MAD.378）；W2服务10.3163us，read交互-.008GB/s、write-.759GB/s。
M96 W13/W2服务交互1.961/2.006us，write残差交互接近零且MAD较大。
将残差按FG调用率换为每调用量后，M180 W13/W2 write交互约-.0083/-.00005MiB；
单位选择会受各状态调用率差影响，不能把GB/s交互解释成独立字节预算。
这说明正服务交互与正流量交互不等价，不能用一个额外正惩罚同时表述需求与响应。

第二场PMU仍在exec46303原串行采集；没有新拟合、额外基准或默认变更。等第二场再确定
状态效应的重复性，所有第一场样本/指标保留，不提前建立通用请求放大公式。


随后exec46303 session2 complete/COMPLETE/exit0，全部3348格完成。第二场rsync81427
明确exit0后，本地analyze.py/interactions.py session2通过完整1116格及31pair检查。
M180 W13 n3 IO1/1、1/32、32/1、32/32第二场write残差.024/2.953/.020/1.803GB/s，
read残差-1.711/2.726/-1.085/5.345GB/s，服务增量-.193/2.219/1.499/15.557us。
M96同顺序write残差.024/2.272/.024/2.238GB/s。额外写主要伴随输出轮换的方向重复，
仍不唯一归因FG写回/RFO或某级cache。M180 W13/W2服务交互11.689/10.216us，
write残差交互-1.231/-1.191GB/s，不能用同一个正混合项解释时间和流量。
第二场PMU/control28格比较保存在session2_pmu_control.json，所有原始JSONL/分析已收回。
全部采集/传输/评分终止，没有活跃基准；不据此自动拟合通用需求放大函数。
下一模型工作应显式区分输入与输出访问历史、读写需求与服务反馈；工作集与cache机制
仍需适用范围验证，当前稳态数据不替代真实路径baseline。默认模型/planner未更改。


## 冻结双角色参数迁移IO状态：热复用前台被明显高估

2026-09-12新增gather_self_pressure/transfer_io.py，保持此前coupled_prediction的h与gamma不变，
只输入新IO状态各自solo流量/耗时，联合数据只用于评分。参数在新IO采集前已拟合；具体预测
在采集后生成，因此不称预注册数值预测。竞争格数由完整grid加载并保留io_transfer.json逐条记录。
首次运行因旧measure模块缓存使六维grid校验失败，未生成结果；修正导入顺序并断言grid的
源码路径为gather_demand_states/measure.py后完整运行通过，Ruff通过。未修改数据或参数。
近零负idle估计仅在物理solver输入截0，原四个solo读写值保留样本，不把负数当物理需求。

按IO状态分组增量MAE：1/1为12.124us，1/32为2.657us，32/1为3.047us，32/32为1.924us。
组内M集合不同（M1718仅1/1与32/32），不将分组MAE直接解释成状态因果效应。
首场热1/1：M96 W13实测-.066us/预测2.641us；M180实测-.216/预测4.935us；
M1718实测+1.269us/预测+48.332us。该模型把全部前台成本按资源因子放大，在热复用状态失败。
背景流量虽然很大，接收方并不一定对其敏感，不能只靠自身负载归一化消除此问题。

第一场无背景worker时间：M96 1/1/1-32/32-1/32-32为13.381/13.458/14.228/15.132us；
M18025.029/25.962/26.782/38.561us；M1718热240.294us、32/32冷408.759us。
热IO参考可用于检验“只放大可受共享资源影响部分”的结构，但热冷差并非已分离的纯计算/
DRAM成本，地址集合/缓存/重叠同时变化；不能直接把差值当作物理访存占比或统一敏感度。
下一结构对照需保持各状态原T0，明确受影响部分的假设，并与状态相关需求预算同时核验。
本轮无新采集或参数拟合，双角色模型仍不推广，默认planner不变。


## 热参考差额响应无重拟合对照：热状态改善，冷状态失效

2026-09-12 M类Lab新增gather_self_pressure/exposed.py。保持h/gamma及各状态原T0不变，
读取同形状、同round的16worker热IO1/1 reference：
`T_i(P)=T_i0+max(0,T_i0-T_ihot)*max(0,h(P)/h(P0)-1)`。
前台cycle取max_i(T_i)+原cycle余项，仍与背景流量反馈自洽求解；没有增加或拟合新参数。
每个worker分别计算，避免用均值代表tail；原请求预算仍固定。负的冷-热差只把暴露分量
截到0，不改变原T_i0。无背景恢复原16worker成本、热reference相等时不变检查通过，Ruff通过。

既有IO状态迁移：增量MAE（原全成本→差额版，us）1/1为12.124→.338，
1/32为2.657→1.015，32/1为3.047→.547，32/32为1.924→18.516。
1/1零响应是构造保证，不是独立泛化成功；各状态组的M集合也不相同，保留逐格数据。
首场W13 n3、32/32：M96预测.390us对实测2.908；M1804.258对15.068；
M171861.416对128.389us。exposed_prediction.json保留时间和读写残差，未只报告热组改善。

该无重拟合版本不推广。原h是在全部成本响应下拟合的，换为差额响应不重新标定的对照
只说明不能直接替换，不能证明所有部分响应模型都错误。同时热冷差混合缓存/地址/重叠
效应，不能直接解释为独立访存时间：若计算与访存重叠，热冷总时间之差可小于实际访存服务。
下一结构工作需要区分重叠和状态新增请求，而非继续把差值当物理计算/访存占比。
本轮无新远端采集、T0修改或默认planner切换，整体goal未完成。


## 2026-09-13：请求服务与T0重叠候选，回顾时间改善但流量仍有缺口

M类Lab新增gather_overlap/run.py，design.json采前冻结64点网格：kr/kw∈
{0,200,400,600,800,1200,1600,2400} effective us/MiB。仅用IO/session1 W13 n3三格
M96 IO32/32、M180 IO1/32、M1718 IO32/32标定；其余33格评价。h/gamma保持原值。
按solo每调用总read/write预算、gather_work逻辑byte份额分配到16worker（不是实测逐worker miss），
`L_i=min(T_i0,kr*Qri+kw*Qwi)`，`T_i=max(T_i0,L_i*max(1,h(P)/h(P0)))`。
联合cycle取max_i(T_i)+原余项，与背景group反馈一起求P。不把热冷差作为访存时间，
也不修改任何状态原T0；latent上限是基线约束，不是物理访存时间的已证估计。
无背景恢复成本、零latent需求时保持原T0、逻辑需求份额和为1检查通过；Ruff通过。

有限网格选kr=1200、kw=200，训练三格MAE1.003us。系数是有效服务参数，不是硬件带宽。
同一33非训练格比较（comparison.json）：
| 版本 | 增量MAE/us | 总read MAE/GBs | 总write MAE/GBs |
|---|---:|---:|---:|
| 整体资源放大 | 5.4618 | 2.0647 | 1.0072 |
| 热冷差额响应 | 3.9275 | 2.0295 | .9420 |
| 请求服务重叠 | 1.2163 | 2.0775 | .9739 |
每个版本严格使用相同排除集；不是拿不同状态集合的汇总指标比较。所有数据此前已检查，
这是回顾性模型选择，新增2参数也需后续独立验证，不能称同复杂度或新鲜泛化胜出。

残余例：M180 IO32/1 W13两场预测5.939/5.916us、实测1.468/1.499us；
M180 IO32/32 W2第一场预测9.170us、实测13.184us。总流量没有同步改善，固定solo预算和
逐worker分配假设仍未解决。results.json保存全部64试验点、逐格时间与流量误差。
候选有条件保留，不替换默认模型或planner；下一步需未标定的中间IO复用状态，以及再到
真实计划的迁移。当前只覆盖此独立16T稳态探针，不宣称完整cost model可靠泛化已完成。
本轮无新远端采集或默认变更。


## 2026-09-13中间IO复用状态验证：参数冻结、独立profile先行

新增E类gather_overlap_holdout，design.json先冻结既有overlap结果/solver和coupled参数SHA，
M96/180新IO状态8/8、16/16、1/16、16/1，1/1与32/32为端点控制。
每状态none或3x8T M12/W13 B32，加idle/BGsolo共26格。预设两场新状态每形状增量MAE<=2us、
max<=5us，同时保留全成本对照和总流量残差；端点不是新状态证据，不据新竞争结果调参。

前台gather_service.cpp只将copy参数校验从1/32扩为1..32的2次幂，现有数组容量32、64call
lead-in、packing/tail/worker源逻辑不变；实际新测试只覆盖8与16，不宣称未测所有参数已验收。
独立编译匹配O2实现TU与O2/SVE256 driver，exec23370 exit0；背景复用原0f383844二进制，
默认扩展不变。新构建布局可能变化，保留端点控制和独立T0校准，不假定二进制计时等价。
measure.py增加--solo-only，仅选择idle/BGsolo/前台solo共14格，完整网格仍26格。
analyze.py检查metadata grid与声明的solo/full集合严格一致，Ruff通过。

执行无PMU/64event各一轮26格烟测（seed612520、rounds1/warmup0），exec28499当前活跃。
Arm NUMA3/controller240、前台16T304..319、背景3x8T280..303，原H4096/SVE256/route96，
背景stage8/4MiB、实际owner1MiB/512KiB；输入输出物理分配容量不变，只改变活动copy数。
下一步仅采14格solo profile，形成冻结数值预测后才启动正式竞争场，避免新joint时长参与
参数或profile构造。烟测只用于执行有效性，不用于选择模型或系数。
当前没有该新状态的正式结果或默认模型修改。

随后exec28499 exit0，两轮26格无PMU/64event烟测均通过。收回数据及binary身份后，analyze.py smoke/smoke_pmu完整校验通过；仅保存结果，不用烟测时间调参。启动--solo-only profile（seed612521、31round/5warmup、14格共504格），exec20680活跃，原64event/100ms协议。数值预测冻结和正式竞争采集仍未开始。


## 中间IO独立profile完成、数值预测冻结后启动正式验证

2026-09-13 exec20680 exit0，14格×36=504格solo profile完成。等rsync43923 exit0后，
本地analyze.py solo_profile通过seed612521、metadata声明14格、数值/CPU/64event及运行率校验。
freeze.py先检查参数文件SHA与design一致、正式control/session文件尚不存在、grid函数来源
为本目录measure.py，再读取31个独立profile样本形成12状态×2模型预测。Ruff通过。
仅solo数据作为输入：16worker成本、调用率、read/write及背景solo，未使用新joint时长或流量。
冻结保留31个相关profile样本和raw rates，负噪声估计仅在物理输入裁0；没有参数重拟合。

冻结overlap预期增量(us)：M96 IO8/8=0、16/16=1.8956、1/16=0、16/1=2.7856；
M180对应0/6.3012/0/5.8497。端点M96 1/1=0、32/32=3.1722，M1800/11.1678。
同时保存full-cost对照和绝对worker服务、总read/write预测，端点不纳入新状态通过率。
具体预测保存在frozen_predictions.json，参数和来源hash一并保存。

独立profile终止且预测冻结后执行
`bash tmp/joint_cost_model_20260911/gather_overlap_holdout/run.sh`，exec80477当前活跃。
脚本记录prediction_launch_identity.txt，再依次control无PMU612541、PMU612541/612542，
每场26×(5warmup+31round)=936格，共2808格，100ms，全部串行，文件存在拒绝覆盖。
默认扩展不变，使用本目录经烟测前台binary与原背景0f383844；NUMA3/16T304..319/
3x8T280..303、H4096/真实route96、W13 B32/full stripes/Ntile16/SVE256保持。
每场新状态按形状增量MAE<=2us且max<=5us的采前门槛保持，流量和全成本对照另报。
暂无正式竞争结果，不调整预测或模型参数；完整planner目标仍未完成。


## 中间IO冻结预测控制场完成：M180的16/1响应高估

2026-09-13新增gather_overlap_holdout/evaluate.py，评分前验证全部frozen source/parameter
SHA及正式seed/grid/31round；不调用solver或更新预测。当前n0只用于配对增量评分与独立
profile漂移诊断，流量只在PMU场评分。Ruff通过。分别保存新状态、端点、absolute service、
profile baseline error与总read/write误差，不混合通过率。

exec80477输出control complete；等待rsync32466 exit0后，本地analyze.py control和
 evaluate.py control通过936格、binary/CPU/grid完整性。冻结时间门槛下：
M96新状态overlap MAE .9819us、max2.6100us（full-cost MAE2.5047），控制场通过；
M180 MAE2.2176us、max6.0260us（full-cost MAE4.2768），控制场未通过。
M180 IO16/1实际配对增量-.17625us，对冻结5.84971us；16/16实际4.2925，对6.30118us；
8/8与1/16实际.4181/.4175us，对预测0/0。M96 16/1实际.1756，对2.78564us，
16/16实际.69875，对1.89558us，8/8与1/16接近零。
这些控制结果不是最终两场PMU验证结论，不调参数或修改门槛来适配本点。

当前exec80477继续session1/2 PMU原串行脚本。控制结果提示solo总byte到latent服务的统一
映射在中间状态仍有高估风险，尤其不能把有off-core流量等同于整个前台受带宽/延迟限制。
需结合冻结流量预测评分检查，但不在本验证集重拟合。默认模型与planner保持。


随后exec80477输出session1/session2 complete/COMPLETE/exit0，正式2808格全部完成。
等两场rsync17718 exit0后，本地analyze.py session1/2和evaluate.py session1/2均通过
完整936格、来源/参数SHA、seed、数值/CPU/64event与31pair校验。未重新生成预测。
新状态M96 overlap增量MAE .9698/1.0189us、max2.5875/2.6550us，两场通过；
M180 MAE2.2126/2.2173us、max6.0435/6.1416us，两场未通过。full-cost同集合MAE
M96为2.4787/2.5224us，M1804.3462/4.3731us；改善不代替冻结绝对门槛。
M18016/1实际-.19375/-.291875us，对冻结5.84971us；16/16实际4.18125/4.21625us，
对6.30118us。M9616/1实际.198125/.130625us，对2.78564us；负/近零控制值全部保留。
所有absolute service、独立profile漂移与总流量误差也在session1/2_evaluation.json中保留。
decision.json为not_promoted：当前重叠结构较全成本对照改善，但不能作为通用响应采用；
不能只选通过的M96重定义整个目标。中间input-only复用下的latent服务映射仍有明显高估。
全部采集/传输/评分已结束，默认模型/planner不变；不在本验证集直接重拟合后宣称泛化。


## 2026-09-13：16/1失败的profile分离诊断

只读核对M180/16-1：独立solo_profile read13.255GB/s，而验证session1/2的同条件n0
read4.761/4.133GB/s；worker服务26.571/26.923/26.956us变化较小。背景solo read
132.517/131.694/131.904GB/s也远小于FG差异。input16_profile_audit.json保留三组原值。
input16_read_repeat.json显示profile first10/last10中位数13.233/12.999GB/s，验证第一场
4.741/4.584、第二场3.710/4.314；不是明显的简单首尾热身趋势。三者min/max均含低/高读量
样本，profile MAD1.703、验证.768/1.361GB/s；不据此唯一归因为cache或运行顺序。

新增profile_diagnostic.py（Ruff通过），对M96/180的16/1分别做四臂：冻结输入、只替换同场
n0每调用需求、只替换n0成本/cycle、两者均替换。h/gamma/kr/kw及BG profile均保持冻结。
换cycle时按调用率重缩放原每调用预算，避免混入需求改变；未用joint压力作输入。
四个冻结输入臂均重现原冻结预测到1e-9us，结果另存profile_diagnostic.json，不改原评估。
M180需求-only预测.519/0us，误差.713/.292us，对原误差6.043/6.142us；成本-only
预测5.904/5.916us，几乎不改善。M96需求-only预测0，误差-.198/-.131us；成本-only仍约2.8us。
这说明当前模型输出强烈受需求profile迁移影响，不能把本次失败只解释成T0估计问题。

该替换使用验证场次的solo数据，是事后条件诊断，不能把原前瞻失败改判通过。
它也不能唯一证明“只需修profile”：off-core总量与关键路径服务的映射本身仍有假设，
小比例成本变化伴随较大流量变化也应保留。下一工作先核验需求profile的状态匹配、稳定性
及不确定性，同时保留latent映射限制；不直接回调系数来拟合本验证集。
无新远端采集、参数更新或默认planner变化。


## 2026-09-13：前序状态关联审计与有界conditioning对照

history_audit.py按原JSONL实际顺序保留前一cell及最近foreground状态，31个M180/16-1
n0样本逐场分组，结果history_audit.json。Ruff初始分号样式问题已修正并通过。
验证session1前一cell有BG的18样本read中位数4.385GB/s，无BG的13样本8.902GB/s；
session2为14样本3.729与17样本5.464GB/s。独立profile31目标点前一cell均无BG，
read中位数13.255GB/s。其它IO/M分组稀疏且混杂，不能据此因果拟合或删除异常点。

新增E类gather_profile_history，复用新copy-count前台与原BG二进制，无native改动。
每次目标测量前执行100ms idle或3x8T M12/W13 B32 prelude；BG DONE后ACK并读取最终JSON，
该JSON在native cleanup/join之后产生，再发送foreground命令，故测量窗口没有活跃BG。
foreground仍执行原输出初始化与64call lead-in。活跃prelude还含启动/validation，实际总时长
并非与100ms idle严格相等，monotonic prelude起止与native结果保留；不称纯cache处理。
M96/180 IO16/1及32/32各两history，再加history匹配idle计数器，共10格，design.json冻结。
至多control及两PMU正式重复，若不能区分则保留不充分结果，不用新数据重拟合模型。

measure.py/validate_smoke.py Ruff通过，校验器检查history、prelude数值/CPU、目标grid及计数。
启动无PMU及64event各10格单轮烟测，seed612560、rounds1/warmup0，exec3522当前活跃。
Arm NUMA3/前台16T304..319、conditioning3x8T280..303/controller240，原H4096/SVE256/
真实route96，背景8/4MiB几何不变。尚无烟测或正式历史效应结果，默认模型/planner未改。

随后exec3522 exit0，两轮10格无PMU/64event烟测均通过。启动run.sh正式control612561、PMU612561/612562，每场10×36=360格，合计1080格；exec12983活跃，全部串行。协议保留conditioning完全结束后才启动目标，不改变模型参数。


## 2026-09-13：history控制与第一场PMU完成，方向随IO状态不同

新增gather_profile_history/analyze.py，检查完整grid、二进制、prelude数值/CPU/teams、
目标窗口仅fg角色、计数器64event；各history使用自己的idle，按同round报告GB/s与MiB/call。
烟测数据完整解析通过，Ruff通过。Native prelude结果在worker join后产生的顺序依据保持；
未将其称作OS调度trace证明。每组允许负idle估计，不做模型拟合。
exec12983已输出control complete/session1 complete；等待rsync79881 exit0后，两场各360格
分析通过。control M18032/32 history3-history0服务增量2.283us（MAD1.185），
16/1仅.06875us；其它小状态变化也较小。

第一场PMU配对history3-history0：M96 16/1 read-2.620GB/s、-.12725MiB/call，
服务+.04625us；M18016/1 read-3.725GB/s（MAD3.214）、-.23027MiB/call（MAD.19581），
服务+.090us。M18032/32 read+3.941GB/s、+.37538MiB/call，write+.925GB/s、
+.10596MiB/call，服务+2.269us（MAD.8975）。M9632/32 read+.183GB/s，服务+.0294us。
前置活动对需求的影响不是统一加/减系数，且16/1流量变化可伴随很小时间变化。

活跃prelude总时长相对idle多约32.17–32.23ms，包含native准备、64call BG lead-in、
结束与校验，已经记录在prelude_us。结果只能归于完整conditioning流程，尚未排除额外
等待时间、启动过程与其它状态因素，不唯一归因为cache或前一个GEMM。
第二场仍在原exec12983中串行继续；不追加样本、不改条件或参数。


随后exec12983 session2 complete/COMPLETE/exit0，全部1080格完成。等rsync96511 exit0后，
第二场analyze.py通过360格完整校验。M9616/1 read变化-2.770GB/s（MAD.815），
-.13163MiB/call（MAD.04043），worker+.0844us；M18016/1 read-3.444GB/s（MAD4.021），
-.20709MiB/call（MAD.23771），worker+.0956us。前者方向/量级重复，后者分散较大。
M18032/32第二场read+.585GB/s（MAD.590）、worker+.483us，明显小于首场+3.941GB/s/
+2.269us；不能把首场幅度当固定历史修正。前置总时长差仍约32.23–32.28ms。
按预设最多两PMU场结束该conditioning诊断，不追加样本追逐特定状态。结果支持前置流程
影响需求profile，但时长/启动/缓存等未分离，不能唯一解释原校准漂移或提取通用历史系数。
所有采集与分析终止；需求profile的条件与不确定性仍需在模型中明确，默认模型/planner不变。

## 2026-09-13：完整计划口径与既有顺序集合审计

按照可靠泛化计划，先核验state_order_holdout能否用于完整计划评分。13个plan均224任务，
同一nested route identity；12个顺序变体共享185个8T/39个16T任务，anchor为182个8T/42个16T。
expert211/245/12在变体由16T改8T，所以anchor不是纯顺序对照。变体有15任务joint cohort。
现有compact仅保留15任务，旧evaluation约2ms的actual是cohort完成时间，不能作为全计划结束。
现有full_baseline_candidate.service逐一查询anchor/control_forward全部224形状，无不支持项。

只读取Arm-codex-internal原state_order_holdout/session1.trace与session2.trace，无新benchmark。
提取脚本及结果保留tmp/joint_cost_model_20260911/plan_completion_audit/；远端同相对目录。
extract_snapshot.py使用本地task_envelopes函数快照，按scheduled_compute起点归一化，
取全部W2末端为compute_end。检查原trace SHA、frontier SHA、correctness、连续call/phase数、
逐任务三阶段worker集合及early_merge=false。两场各533call/403正式plan样本提取完成。
前两次命令分别因远端缺分析模块、快照多截一个空函数定义在读trace前退出；修正后一次完成。
rsync27249 exit0后本地analyze.py核对全部旧compact任务、native e2e和两场完整anchor任务
逐字段相同，并检查224任务依赖端点及各plan正式pair5..35完整。Python分析及Ruff通过。

原协议保持：Arm NUMA3/CPU240..319，H4096/F512/BF16/SVE256/Ntile16，2048token/12288route，
4权重副本、216MiB scrub、pretouched workspace、5warmup+31正式配对，seed611681及原session2。
8/16T均(t,0,0,1,1)full stripes，W13/W2总8/4MiB；8T owner1MiB/512KiB，16T512KiB/256KiB。
二进制及校准身份沿用原session JSON，不以当前工作树替代历史构建身份。

完整计算结果：anchor28697.71/28642.22us。12变体第一场最小new_09=29536.29、
最大control_reverse=29752.45us，范围216.16us；第二场最小new_04=29498.37、
最大control_forward=29776.00us，范围277.63us。各变体MAD范围90.55..271.01/
100.43..277.23us。局部cohort最快为new_04/new_01，其全计划候选集regret为.723/.398%。
66个变体pair的配对中位数差只有30对跨场方向一致；这只是描述性符号统计，不是显著性检验。
results.json保留全计划/局部排名、逐lane完成和末任务；pair_consistency.json保留全部配对差。

结论：现有数据可做历史完整计划诊断，但不能把局部排名当完整排名，也不能以单场赢家作为
稳定选择证据。这不是新模型优劣结论，尚未对12变体重放完整候选模型。后续先做同集合完整
模型评分与关键路径误差，再以新路由/更可分辨计划作冻结验证；不因本集合差异小放宽门槛。
无模型参数、production默认或planner行为变更；可靠泛化目标及等预算搜索对照仍未完成。

## 完整顺序集合模型重放启动与初步路径诊断

新增plan_completion_audit/replay.py、score_replay.py，E类Lab分析入口，无native/默认/公式修改。
impact-analysis、test-selector、code-review-gate按当前分析范围执行；CodeGraph未定位tmp实现，
直接核对baseline.service、segments.legacy和冻结simulate.predict。按每个bridge重建全部
224任务的width、CPU、依赖/后继和成本；窗口和总route数严格检查。无实测事件作为预测输入。
使用原31组独立initial_entry情景、固定setup/gap/publication和已有成本/响应参数，三臂为
zero、original(family8/local16)、linear gather；此前失败的quadratic不作为本轮候选。

命令`.venv/bin/python tmp/joint_cost_model_20260911/plan_completion_audit/replay.py --smoke`
完成anchor/control_forward各三臂单情景。anchor重构jobs逐字段等于legacy_jobs，三臂所有
阶段端点相对历史预测<1e-7us。两脚本Ruff通过；score.path_error在两plan样本上验证路径
分项之和严格重现总误差(<1e-6us)。无相关kernel修改，未跑native正确性或新benchmark。

完整命令去掉--smoke，exec71899运行13plan×3臂×31情景，预计约15分钟本地CPU。
逐plan/variant以独立JSON保存于replay_full，未完成全39文件前score_replay拒绝正式评分。
anchor31情景已完成：zero27148.855、original27482.750、linear27665.478us，均复现历史。
control_forward三臂已完成：28018.510/28380.791/28560.618us。

initial_path_diagnostic.json记录前两plan的事后路径分解：每个实测call回溯最晚依赖路径，
预测采用31情景mean端点，按stage/width、ready gap、internal gap及末任务差逐项记账。
每条记录代数闭合；平均分项不是median误差分解，也不构成因果归因。
anchor original两场16T/gather误差-550.1/-557.7us、W13-182.2/-59.7、W2-408.9/-445.4，
ready gap-63.0/-102.8。linear主要把gather改为-366.6/-374.1，其余基本不变。
control_forward original的8T/gather为-349.9/-377.1、W13-605.1/-665.2、W2-273.5/-320.5us；
linear的8T/W13仍-606.6/-666.9，不能由加强gather响应消除。
control_forward_contributors.json进一步保存逐expert路径加权误差：expert59/M529/8T
两场出现在26/29条路径，gather贡献-227.0/-258.4us、W13-149.2/-164.2；M13/15/17等
小中M的8T W13亦有累计低估。说明原anchor的16T误差结论不能代表所有计划关键路径。

当前仅前两完整plan的阶段诊断，不是12变体最终误差/选择结论，不调整任何模型参数。
完整exec仍活跃；完成后运行score_replay.py，检查全部lane和候选集regret，再更新本记录。

## 8T路径低估的独立基线交叉检查

继续poll exec71899确认运行，未重启；完整评分尚待全部39文件完成。等待期间只分析已有
shape_grid_batched train_01/02/03两场compact，核对frontier SHA、isolation/correctness、
31pair完整、实际8T和两stage full-stripe窗口。当前full_baseline_candidate.service与这些
历史独立expert比较，记录plan_completion_audit/isolated8_baseline_audit.json。
M1..12 W13/W2 MAE .8375/.5017us；M13..24为1.3855/.5353us，bias+.459/+.278us；
M36/48/60为2.0583/.7417us。M13/15/17的W13基线误差分别+2.44..4.15、+.06..1.14、
+1.64..1.84us。该历史cross-expert检查不是新holdout，不能视为计划内状态严格匹配。

large8_baseline/validation中已有iso_m0529 expert85与当前预测比较，W13误差
-12.694/-9.814us，W2-3.158/-1.218us，保存isolated529_audit.json。真实计划M529是expert59，
故这里只说明同M基线的已有验证量级，不把它当expert59相同历史下的独立真值。

control_forward_response_gap.json保存全部8T任务在两场中的预测stage增量及实测高于冻结
T0的差额。W13例：expert59/M529预测增量20.61us、实际差额198.75/194.81us；
expert168/M13预测13.15、实际57.21/64.51；expert172/M15预测10.01、实际56.73/57.53；
expert224/M17预测11.47、实际41.52/43.13。实测差额仍混合竞争、状态迁移和同步，不能
直接命名为纯竞争slowdown，也不能据此拟合case系数。

当前证据使下一优先级转向竞争响应的作用阶段/请求时序，同时保留状态迁移检查。
现有explicit legacy segments只在first段产生请求，后段read/write=0且8T sensitivity=0；
这与需要验证的完整块/尾块竞争响应存在结构缺口。暂不扩大T0或添加统一team惩罚。
以上均为既有数据诊断，无新采集、无参数修改；完整13plan重放继续运行。

## 完整13-plan重放结束与统一评分

exec71899 exit0，13plan×3模型×31情景全部完成。score_replay.py补充冻结source SHA、
39文件唯一覆盖、任务集合和各情景completion=max(W2 end)校验；exec85192 exit0，
结果replay_scores.json。两脚本Ruff通过，全部anchor端点等价检查及路径误差代数闭合通过。
无新Arm测试，历史数据协议和限制同前；本次独立入口情景不与实测call强行配对。

| 12顺序变体 | 第一场MAE/us | 第二场MAE/us | 平均相对误差/%(两场) | 预测选择 | 实测候选集regret/%(两场) |
|---|---:|---:|---|---|---|
| zero |1627.858|1642.220|-5.490/-5.536|control_forward|.388/.941|
| original |1267.951|1282.313|-4.276/-4.323|new_09|0/.576|
| linear gather |1088.538|1102.900|-3.671/-3.718|new_06|.126/.809|

linear选中计划实测29573.50/29737.07us，original为29536.29/29668.31us。
历史同集合绝对时间改善不等于选择改善；之前跨场噪声和排名不稳定限制仍适用。
original预测12变体范围仅11.209us，linear9.246us；不把实测216/278us范围全解释为
模型丢失的顺序效应，因为该范围包含显著的测量波动。

12变体的平均观测路径记账：original8T/W13误差-614.9/-631.9us、W2-308.0/-306.2，
gather-351.0/-375.4。linear对应W13-616.1/-633.6、W2-308.2/-306.0、gather-254.7/-273.0。
这些均为mean端点/mean路径分量，不与median MAE混加，不是因果瓶颈证明。
因此8T后续GEMM响应缺口在整组计划中重复，而非只在control_forward出现。

随后按next_ablation_design.json准备ablate_segments.py并更新数学模型Lab语义：两个plan，
请求时序首段/均匀与8T后段敏感度零/首段值交叉，所有T0、总Q、16T敏感度和拟合系数固定。
四臂transform守恒检查通过，执行前要求replay_scores存在。首次运行在原臂零响应检查中
遇到JSON字符串expert键与内存整数键不一致，未产生arm文件即退出；改为str(expert)，
Ruff通过后重启一次。当前exec33218活跃，anchor原臂复用冻结31情景27482.750us，
其零响应端点校验通过；其它臂继续计算。无新模型采用或泛化结论。

## 固定预算2×2完成：后段响应对8T路径有明显影响

exec33218 exit0，两个plan×四条件全部31情景完成（原臂复用已冻结预测），八臂逐阶段
T0/总read/write守恒、零响应首情景全部阶段端点<1e-7us检查通过。score_ablation.py
检查八臂唯一覆盖、系数相同、31情景和守恒后，exec61440 exit0；Ruff通过。
结果segment_ablation_scores.json，所有预测和jobs在segment_ablation/。
本地纯模拟，无新Arm采集；参数未拟合，原始数据与协议同完整plan审计。

| plan | 原预测/us | 只改时序/us | 只改后段响应/us | 同时改变/us |
|---|---:|---:|---:|---:|
| anchor |27482.750|27487.730|27577.260|27587.665|
| control_forward |28380.791|28380.108|28866.895|28858.704|

相对原预测，anchor时序/响应主效应+4.979/+94.510us、交互+5.426；control_forward
对应-.683/+486.104us、交互-7.508。control_forward只开放后段响应后，两场总时间误差
由-1269.999/-1395.209us缩至-783.895/-909.105us；anchor仍-1120.450/-1064.960us。
control_forward观测路径平均8T/W13误差由-605.1/-665.2变为-249.3/-286.0us，W2由
-273.5/-320.5变为-157.1/-195.9；gather约-350/-377us基本不变。这里是mean路径分项，
不与median总时间误差相加，也不当因果分解。

结论限于本诊断：后段零响应假设限制了8T计划预测，值得独立标定；在旧总请求预算下，
这两种时序假设之间的差异较小，不能证明真实时序不重要或旧总预算正确。
不采用“所有后段等于首段敏感度”，不以两个历史plan改善作为泛化验收。后续保持T0，
在已有可控cohort上校准后段响应并分离同域/跨域，再用未用于拟合的动态条件验证。
当前所有模拟已结束，无活跃采集/传输句柄；默认planner和生产模型不变。

## 可控8T cohort重放：M12揭示高并发响应缺口

新增controlled_response_audit.json及controlled_overlap_audit.json，读取locality_m12、
locality_spliced(M13)、locality_m48已有两场数据。核验frontier SHA、isolation/correctness，
stage增量按同场local_n0的同pair相减再取median。四背景下W2增量很小，但实测GEMM envelope
仍重叠：local_n4的M12/M13/M48 W2平均活跃背景数中位数约3.9/2.0/1.0，全部31call非零。
因此不能将低W2增量简单解释为无背景；活跃任务数也不是实际访存压力。

controlled_replay.py仅构造初始独立joint cohort，检查无前驱、8T/full stripes/early_merge关，
suffix由原计划保证在cohort结束后执行。复用冻结T0/entry/setup/gap/publication/coefficients；
比较原响应、仅W13后段、仅W2后段、两者开放，共3M×8条件×4臂×31情景。
exec80088 exit0，24文件全部完成；Ruff通过。结果controlled_replay_scores.json，未拟合参数。
历史两场、local/cross、n1/2/4的stage增量MAE(us)：
| M | W13原/两后段 | W2原/两后段 |
|---|---|---|
|12|9.429/9.429|.411/.411|
|13|12.362/11.180|1.225/.571|
|48|6.829/5.956|2.096/2.151|

关键定位：M12没有后段，仍有高并发W13低估。第一场local n1/n2/n4预测增量
3.04/6.13/12.08us，实测2.50/5.97/40.44；cross对应3.03/6.10/12.10，实测2.35/10.15/38.48。
不能靠增加后段敏感度修复，也不支持对所有并发统一放大首段系数。
当前simulate使用线性压力响应的固定点反馈，没有硬约束总请求服务率不超过共享容量；
原200GB/s只是系数归一化尺度，不能称已满足资源容量。下一步先诊断这一饱和机制，
再校准后段响应；不直接增加shape/team经验惩罚。此结论不唯一证明物理DRAM瓶颈，
旧请求预算和实际路径流量迁移仍有缺口。全部本轮计算已结束，production/default不变。

## 固定共享容量诊断：M12/M13改善，M48仍需区分

新增独立capacity_simulate.py快照，在原pressure固定点每轮对发请求片段的target速度按
min(1,C/sum(rate*target))统一缩放，零请求片段不缩放；每个区间验证实际总rate<=C。
固定C=200000/1024KiB/us，不拟合。请求仍为legacy GEMM/gather逻辑预算，不能称物理DRAM
容量校准；更新MATHEMATICAL_MODEL Lab语义，不改production、ABI、默认planner。

check_capacity.py验证1/3/5同步team解析max(T0,nQ/C)，关闭/极高容量时完整anchor首情景
全部端点<1e-7us，以及0/负/NaN拒绝。首次测试fixture遗漏state和entry使用list，分别在
输入校验处失败；修复fixture后exec47001 exit0，capacity_checks.json保存通过记录，
模拟器/检查/重放Ruff通过。不是通过放宽数值容差或修改解析预期来通过。

replay_capacity.py使用全部24既有可控cohort，容量开启且后段响应零/首段值两臂×31情景。
exec90378 exit0，capacity_replay/及capacity_replay_scores.json完整保存。全部独立n0保持T0。
| M | W13增量MAE：原/容量/容量+后段(us) | W2对应(us) |
|---|---|---|
|12|9.429/3.148/3.148|.411/.264/.264|
|13|12.362/1.775/2.468|1.225/1.225/.571|
|48|6.829/7.379/9.420|2.096/2.096/2.151|
M12 local_n4预测W13增量由12.075变44.912us，实测两场40.44/37.71；低并发n1/n2不变。
M13高并发约45.5us更接近实测；M48容量臂约45.3us反而高于实测26.85..31.25us。
因此只接受容量机制值得继续研究，不采用固定200值/统一后段敏感度，也不称可靠泛化通过。

full_capacity.py随后将相同C和两个后段条件迁移到anchor/control_forward，固定全部T0、
原coefficients和31entry情景，不用实测事件作为输入。Ruff通过，exec76170已启动，
四个full-plan结果尚未完成。与既有无cap四臂比较以检查总时间影响，无新Arm采集。

随后exec76170 exit0，四个full-plan×31情景全部结束；full_capacity_scores.json保留两场
完成时间及逐stage路径分解，代数闭合校验通过。容量单独开启：anchor29261.321us，
误差+563.611/+619.101us(+1.964/+2.161%)；control_forward30022.113us，误差
+371.323/+246.113us(+1.252/+.827%)。同时开放8T后段响应：anchor29422.472us，
误差+724.762/+780.252；control_forward30358.666us，误差+707.876/+582.666。
总时间绝对误差小于原模型，但高估方向一致，不能视为已完成泛化或据此调C。
这也说明无容量版本中诊断有效的后段响应不能独立叠加后直接采用。
下一步C只从独立可控竞争条件标定，再检查M/放置和完整计划迁移；M48的回归与旧预算
时序限制必须保留。全部本轮计算已结束，无活跃句柄，默认planner未改。

## 单条件容量标定与新M验证准备

fit_capacity.py只以历史M12/cross_n4/session1 W13配对增量38.48us为目标，容量区间150..300
等效GB/s作16次二分，所有T0、请求预算、原响应系数和后段零敏感度保持。每次31个独立entry
情景，预测n4减n0后取median；未用full-plan时间拟合。exec1674 exit0，结果
capacity_fitted.json：C=206.8645477294922，预测增量38.479349647us。Ruff通过。
数据此前已看过，标为development calibration，不把此单点拟合误差当泛化证据。

validate_capacity_fit.py核对冻结来源SHA，重放24个历史cohort和两个完整plan；exec53837
exit0，26文件完整，结果capacity_fitted_replay/和capacity_fitted_scores.json。
排除唯一训练条件后的stage增量MAE：M12 W13/W2=1.449/.264us，M13=3.448/1.225，
M48=5.232/2.096。M12统计排除了训练点，不直接与之前含该点的聚合MAE比较。
完整anchor29088.960us，对两场+1.363/+1.560%；control_forward29832.188us，
对两场+.612/+.189%。这些仍是历史迁移检查，未调参数匹配完整计划。

新capacity_holdout_m17、capacity_holdout_m35复制已验证M48 cohort bridge并只改变前台
route_prefix/expected rows和nested sizes，background仍48/24/12/12、local/cross n0/1/2/4，
每组八条件加anchor，8T full stripes。design.json固定seed613001/2及613021/22，
5warmup+31formal、两场。预设各M/各场/各stage增量MAE<=5us、max<=12us，
且相对原模型MAE回归不超过1us。该验证是新joint M，同route/background，不是新工作负载。
freeze_capacity_holdout.py检查不存在session文件并冻结容量来源，将前台的service、gather
请求和segments全部按新M重建，保留背景和独立entries；Ruff通过，exec74230启动预测冻结。
远端采集尚未启动，先保留数值预测，再执行runtime身份/烟测/正式两场；不可按新结果调门槛。

随后exec74230 exit0，M17/M35各八条件×两模型×31入口场景数值预测均已冻结，
两个frozen_predictions.json存在；当前无活跃计算或采集。待执行远端身份校验和烟测。

## 新M17/M35正确性通过，正式冻结验证启动

Arm-codex-internal读取默认extension SHA dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f，
experiment calibration cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41，
与原实验一致；未见冲突的已知benchmark进程。只同步新两目录，rsync31777 exit0后再运行。
smoke.sh/run.sh由原locality_m48脚本替换目录和已冻结seed，其余协议不变，无native重建。
exec35825串行两组--correctness-only烟测exit0，两组bitwise_correctness=true。
frontier SHA M17=59e6fc1836e77cf83623143a54e9973caa5c9326dcf279e7be343a319360f0b7，
M35=d3433c906aab6e546d21c7d0f5cd92de4df6c72a8060f3ee88002bf68a430c8b；
runner SHA均5da239667ce370ced464100fd7c6f1ac4db097cd0c96c8f8d1ea0ad56a72e6d5。

evaluate_capacity_holdout.py在正式launch前写完并通过Ruff；它只读取冻结预测，不调用solver。
按校准审计约定，实际每pair相对同场local_n0的stage增量取median，预测按同entry相减取median；
两个n0单独保留，六个n1/n2/n4竞争条件用于每stage门槛。校验预测来源SHA、seed、frontier/
扩展/校准身份、369call、31pair及nested target M，门槛保持设计值不变。

记录两预测文件SHA到launch_prediction_identity.txt后，exec92592启动串行
`bash .../capacity_holdout_m17/run.sh && bash .../capacity_holdout_m35/run.sh`。
每M两场，每场9plan×(5warmup+31formal)，另有45正确性prefix call；NUMA3/CPU240..319，
H4096/F512/BF16/SVE256/Ntile16，8T (8,0,0,1,1)，W13/W2总8/4MiB、owner1MiB/512KiB，
4权重copy、216MiB scrub、固定pretouched workspace、early merge关，与历史协议一致。
当前正式采集活跃，尚无新条件最终评分；默认planner不变，不更新预测或参数。

## 新M17/M35四场冻结验证全部通过

exec92592依次输出M17两场validated/COMPLETE、M35两场validated/COMPLETE并exit0，
每场369call完成原trace/compact验证。分别等待rsync35910、74900、81949、39477 exit0后
才运行evaluate_capacity_holdout.py对应M/session，四次均exit0，冻结source/seed/extension/
calibration/frontier/31pair/target M和门槛校验通过。原始trace保留Arm同实验目录，本地
session和compact及evaluation完整；没有重跑、追加采样或改变冻结数值。

| M/场 | W13增量MAE原/容量(us) | 容量最大误差(us) | W2增量MAE原/容量(us) | 容量最大误差(us) |
|---|---|---:|---|---:|
|17/1|10.366/1.434|2.969|.930/.930|2.097|
|17/2|10.665/1.733|3.729|.648/.648|1.107|
|35/1|6.682/3.916|10.335|.622/.622|1.260|
|35/2|7.780/3.571|7.245|.840/.840|1.930|

各M/场/stage均满足MAE<=5us、max<=12us及相对原模型MAE回归<=1us。
capacity_holdout_decision.json为passed_declared_new_m_gate，保留全部四evaluation来源。
次要绝对stage诊断未改变门槛：M17 W13 MAE .873/2.302us、W2 .707/.550；M35 W13
2.966/2.399、W2 .597/.704。结果capacity_holdout_absolute_diagnostics.json。

接受范围：固定206.86455等效逻辑容量在此次同route/同expert/background的未标定joint M17/M35
条件上通过前瞻验证，W13改善、W2保持；不称物理DRAM容量测量，不自动采用到默认planner。
完整工作负载、其它宽度、动态长阶段需求和等预算planner搜索仍未完成；历史M48残差保留。
下一步进入新工作负载完整计划的时间/lane检查，继续保留旧模型同候选对照。
全部采集、传输、评分已结束，无活跃句柄；未提交或修改production默认。

## uniformish完整计划迁移准备

核对原median.json确为当前主要标定route request016/case017 layer4；uniformish.json是
request022/case023/zh2048-024 layer20。实际route SHA核验并bincount得到234活跃expert、
12288route、maxM697：M1..12有91个、M13..60有83个、较大M60个。原anchor全8T，
对全部234形状调用冻结service无unsupported，不需要外推宽度或形状。该route历史上已有
isolated数据被检查，不将其称作完全未看过的数据；本轮是新的完整计划联合测量。

prepare_uniformish.py检查原bridge只有lane chain，identity重排逐字段等于原bridge。
在完全相同lane/expert/width分配下生成anchor、reverse、small_first、staggered四个唯一
bridge；资源依赖、全task元数据/ragged allowed widths和full-stripe geometry保持。
新目录capacity_uniformish_full含frontier/counts/design，代码Ruff通过。
预设两场seed613041/2，逐plan/session完成时间相对误差<=5%；逐lane误差<=max(100us,
实测lane时间的10%)；预测所选plan的实测候选集regret<=2%，噪声内不宣称稳定排名。
计划不改变lane membership，此验证不是等预算完整planner搜索。

freeze_uniformish.py按实际counts和bridge构造全234任务，复用T0、family8响应、gather及
setup/gap/publication和31独立entry，比较原模型和冻结C=206.86455容量模型，后段保持零。
声明来源SHA，禁止读取新session；所有预测完成后才写frozen_identity。Ruff通过，
exec90734已启动4plan×2模型×31情景，当前anchor原模型26745.626us，其余继续。
run/smoke脚本由验证通过的原脚本替换路径/seed、去掉nested-routes及worker-details；
target_experts含全部234任务，用完整trace端点评分。远端同步/烟测/正式测量尚未启动。

随后exec90734 exit0，四plan×两模型×31情景全部完成，frozen_identity.json的八文件SHA
逐一校验通过。预测完成时间(us)：anchor原26745.626/容量27852.100，reverse
26848.455/28045.853，small_first27283.275/31416.037，staggered26848.455/28045.853。
这些是采前预测，尚无真实速度/顺序结论。特别检验容量模型对small_first较大差异的判断。

evaluate_uniformish.py已准备并Ruff通过，核对161call/31pair、全部234任务、预测与输入SHA、
extension/calibration/seed/frontier，按全部W2末端与十lane末端评分；执行预设每plan5%、
每lane max(100us,10%)及候选集regret2%门槛，保留原Lab联合响应对照，不把它混称生产planner。
大预测留在本地，只同步frontier/counts/design/frozen_identity/run/smoke，rsync39737 exit0。
远端确认默认extension和calibration仍为dd554ea.../cdccff46...，exec13947启动四plan
--correctness-only烟测，尚未开始正式采集。无模型或门槛修改。

exec13947随后exit0，smoke.json bitwise_correctness=true，frontier SHA
ac43001e4a32dfd89713b84ccf8e156c0504254c81ad6bd4b0a30a73547e095a，runner5da23966...保持。
记录frozen_identity SHA后exec85436启动run.sh，两场613041/2串行，4plan×36round，
非nested正确性prefix17call，每场预计161call；只用完整234task compact、不含worker明细。
当前正式采集活跃，尚无uniformish新完整计划实测结论。

## uniformish两场完整计划验证通过，搜索覆盖仍不完整

exec85436两场validated/COMPLETE/exit0。分别等rsync46894、7147 exit0后运行
evaluate_uniformish.py 1/2，两次exit0；161call/31pair、全部234task、八预测文件和来源SHA、
runtime/seed/frontier校验通过。原始trace保留Arm同目录，compact/metadata/evaluation本地完整。
decision.json为passed_declared_full_plan_gate，包含两场评分与所有配对时间差分布。

| plan | 容量预测/us | 第一场实测/us | 第二场实测/us | 容量误差/%两场 |
|---|---:|---:|---:|---|
|anchor|27852.100|28079.56|27875.56|-.810/-.084|
|reverse|28045.853|27874.37|28026.58|+.615/+.069|
|small_first|31416.037|30344.17|30539.74|+3.532/+2.869|
|staggered|28045.853|27943.07|28009.20|+.368/+.131|

容量平均绝对误差393.398/238.921us，原Lab联合响应1628.840/1681.317us；容量模型
全部plan/session完成时间5%门槛通过，全部十lane通过max(100us,10%)门槛。
最大lane绝对误差为small_first1194.24/1162.92us，保留而不隐藏。
两模型均选择anchor，其实测候选集regret为.736%/0%，所以不能称选择质量改善。
small_first相对anchor的配对减速median2427.78/2763.34us，P10..P90为
1895.86..2706.07/2221.39..3033.24us；这是配对样本分布，不是置信区间。
容量模型预测差3563.94us，方向正确但幅度偏大。其余近似计划排序跨场改变。

search_coverage_audit.json检查完整混合宽度搜索准备度：本冻结Lab provider对234任务
8T全覆盖，16T覆盖155任务、缺79任务/63个不同M；1/2/4T均无provider实现，各缺103种M。
这是当前新Lab provider的覆盖限制，不是生产planner能力限制。下一步必须补齐并验证这些
成本再做完整等预算搜索；不能把四plan顺序对照缩称完整搜索，也不能伪造缺失成本。
本轮接受固定参数在此all8T完整工作负载的有限迁移结果，保留动态需求/历史/M48残差和
其它宽度验证需求。默认planner不变，所有采集/传输/评分结束，无活跃句柄。

## 混合宽度provider数据审计与旧worker明细恢复

width_provider_audit/inventory.py核对五份历史frontier的十份compact，得到94个独立记录。
extension均dd554ea...，workspace Python源7e012467...、fixed_pretouched_workspace及Ntile16
与当前协议相同；已查目标窗口均0/0(full stripes)。已核验的M覆盖：
1T={1,3,4,7,8,10,16,24,62,714,1341}；2T/4T各{1,3,4,7,8,16,24,62,1341}。
这是已查子集，不是全档案穷尽。旧frontier没有route_file；有routes时直接取，否则保留
validated/report.json的trace/expert唯一M来源，不伪称本轮重新验证了其route tensor。
初稿假设route_file和所有report行均有trace分别失败，改为显式来源和跳过无trace的其它
报告行后inventory完成；Ruff通过。全部94条旧compact均无worker_details。

旧small_t_isolated validated报告complete_model_eligible=false，主要gather特征碰撞和M7
验证失败；不能把现成系数直接接为新provider。新模型需要worker服务，而旧数据只有stage
包络，必须先分开arrival等待。16T已有first-row/bulk和h1/h2/h4/h140六种odd-tail数据，
但h140 W2 r1/3/5/7净增量约-279..-293us，不能作为正尾块服务或任意线性插值。

远端原.log路径不存在；rg不可用后用文件名find定位到四个对应.log.gz归档，未丢失。
新增独立analyze_gzip_workers.py快照，仅把输入读取/哈希改为gzip流式解压，仍校验解压后
SHA与原session trace SHA一致；原归档/旧分析不改。Ruff通过，rsync23618 exit0。
exec80181启动small_1_16及width_2_4各两场原trace的worker-details重解析，使用已验证
locality_grid_runner辅助模块，分析绑定CPU0，无新kernel测试或性能采集。
当前重解析活跃；完成后须先与旧包络逐字段核对，再评估窄team gather服务特征。

随后exec80181 exit0，四份各521call/13plan全部验证通过；rsync89596 exit0后取回worker
明细。逐call检查全部旧tasks和native_e2e字段严格一致，结果worker_gather_audit.json。
1T M7 worker服务12.27/10.71us，M8为21.29/19.82us，单worker故不存在team到达spread解释。
2T M7 mean-worker7.55/7.43us，M8为13.09/12.895；4T对应4.418/4.265与6.675/6.713。
2T/4T另有明显envelope额外时长：例如第一场M7约11.05/31.96us、M8约21.10/41.58us。
额外时长是span减max(worker duration)的每call统计，不称纯同步成本。

因此旧gather拟合失败同时涉及worker服务差异与到达时序，不能只扣掉一个固定等待项。
这些M7/M8来自不同expert/输入，不能唯一归因为kernel形状；下一步窄team基线采集应使用
同一expert的M1..12嵌套前缀并记录worker服务，先隔离输入/布局和历史，再建立形状模型。
四份恢复结果仅重解析旧数据，没有新kernel性能采集。当前无活跃句柄；旧失败系数未接入。

## 同expert窄team M1..12基线采集启动

prepare_narrow_prefixes.py复用isolate_bridge/validate_resources，将原median expert96(M1718)
置为独立head；1/2/4T分别CPU319、318..319、316..319。每宽度M1..12嵌套前缀加原anchor，
其它expert保留并等待目标完成，early merge关闭，显式full-stripe (T,0,0,1,1)。
每宽度W13/W2总8/4MiB，owner分别8/T和4/T MiB。design明确这是发现/标定数据，
session1可标定、session2只重复检查，后续新M/历史点另作验证，不预称泛化。
每组13plan的资源检查及Ruff通过；run/smoke统一使用已验证locality_grid_runner。

默认扩展dd554ea...及实验校准cdccff46...仍一致。rsync35042 exit0后启动exec2939，
三组--correctness-only烟测均exit0且bitwise_correctness=true，runner5da23966...。
frontier SHA：1T4113b2a6a77fb1a2dfa02a1e4c3319df1b14ad0a0d0da648661741f5b77c1ef7；
2T69964d7c93ae8e4f418226cc52acc847075aebdf918f1fba754ea28b8b68bd5d；
4Tc54f6bb020f25952b6c55980ea6206fb0020215876b8ee57203553f4ca9e25a9。

analyze_narrow_prefixes.py已准备并Ruff通过，检查533call/31pair、身份/seed/nested M，
每stage worker CPU/数量、服务正值和包络端点一致，保留逐call worker、mean/max服务及
arrival spread，不将mean-worker与stage median互相替代。
exec11043已启动三宽度各两场串行run.sh，seed613061/2、613081/2、613101/2；
5warmup+31formal、4权重copy/216MiB scrub/固定pretouched workspace/NUMA3 CPU240..319，
H4096/F512/BF16/SVE256/Ntile16沿用原协议。每场13×36加65正确性prefix，预计533call。
当前正式采集活跃，无新拟合或默认provider修改；待完整校验后再分析同输入的M7/M8差异。

## 窄team六场完成：gather分界复现，1T低M GEMM有状态漂移

exec11043三组两场全部validated/COMPLETE并exit0，每场533call。对应rsync45617、57175、
3510、35349、69944、7975全部exit0后，analyze_narrow_prefixes.py六次均通过，得到每场
36个shape-stage记录与全部31pair worker明细。无追加采集/样本删除或失败点替换。

同expert gather mean-worker中位数(us)：1T M7/M8两场12.37/13.84与12.76/14.15，
相比旧跨expert M7/M8约12/21的差距缩小；M9为30.91/30.23，M10约38us，分界转到M9。
2T第一场M7/M8/M9=7.42/8.53/16.68，4T=4.36/4.98/9.04。joint_gather_work镜像在M9
由8行packed tail切换到12行main panel，支持模型至少区分该路径，但不是全部耗时差的
唯一因果证明。4T第一场M8 mean-worker4.98us、stage13.56us，继续独立保留arrival影响。

| width | gather mean-worker重复MAE/us | W13 stage重复MAE/us | W2 stage重复MAE/us |
|---|---:|---:|---:|
|1|.464|11.441|8.115|
|2|.167|2.882|2.286|
|4|.458|.587|.617|

最大差异：1T/M2 W13第一/二场429.99/369.83us、W2=202.61/169.33us；1T/M1也由
432.05/202.11降至401.74/185.09us。first_kernel_repeat_diagnostic.json按copy分组后，
该变化不限于单一权重copy，不唯一归因于cache、频率或kernel；保留全部样本和状态不确定性。
2T最大W13差5.43us、W2差11.87us；4T对应1.55/5.17us。

width_provider_audit/narrow_first_rows_bank.json保存session1的36个(width,M)独立成本：
gather为逐worker中位数，W13/W2为stage envelope中位数，附源SHA及session2重复诊断。
这是发现数据表，不修改原冻结provider，不宣称后续历史/竞争或新M已覆盖；尤其不能因
表存在就把1T/M1/M2当稳定通用成本。后续先建gather分路径成本，再处理低M1T状态漂移与
M>12完整块/尾块基线，目标仍是完整混合宽度搜索而不是只覆盖M1..12。

同时回看16T h140旧worker诊断：M1680第一场stage median5421.79us而median-worker
5115.18us，16/31call有>150us finish spread；第二场15/31call，stage median5134.41。
M1681 median-worker5118.02/5118.33us且无该spread。因此旧-292us尾块差还混合了完成
分布的中位数跳变，不能直接拟合为负物理尾块服务；原诊断阈值为事后描述性，未变成过滤器。
全部本轮采集、传输和分析结束，无活跃句柄。默认planner和原模型不变。

## 窄team gather分路径候选与多panel冻结验证

fit_gather_paths.py用第一场奇数M={1,3,5,7,9,11}的逐worker中位数做非负最小二乘，
每(width,M)总权重相等。对照为统一[1,read_KiB,write_KiB]三系数；候选按packed rows8/12
分成六系数，跨1/2/4T共享。得到8路径(1.416790,.105715,.079857)，12路径
(1.708158,.417754,0)。截距单位us，其余us/KiB；系数不是独立读写带宽，write=0为有效
回归边界，不表示写入免费。数据已经看过，此划分明确为开发检查，不称前瞻holdout。
偶数M两场条件等权MAE由3.132/3.160us降至.688/.778us，最大worker误差由10.634/10.324
降至2.902/2.722us；奇数第二场MAE .444us。结果gather_path_fit.json保留全部记录。

gather_paths.py按gather_work片段计算上述成本，默认只支持M1..12/1,2,4T，显式实验选项
才可组合到M96。单panel168记录端点预测一致<1e-10us，非法/未验证默认范围拒绝；Ruff通过。
多panel时暂逐片段加截距，单panel数据不能区分该启动项是否应每worker只加一次，因此
该组合假设必须验证，未提前接入生产或原冻结provider。

prepare_narrow_history.py创建narrow_history_t1/t2/t4，固定同expert96前缀，M12控制以及
13/15/17/19/21/23/24/36/48/60/96，维持完整条带/原隔离bridge。gather shared/packing两
版本数值在采集前冻结，源/前沿/设计SHA保留；GEMM单独声明为完整块与h1尾块标定数据。
gather新M门槛：每width/session条件等权worker相对MAE<=10%，每worker绝对误差
<=max(3us,实测15%)，M12控制不纳入新M门槛。evaluate_gather_history.py已写完并Ruff通过。
analyze_narrow_prefixes增加显式--history以读取声明形状，其余身份/worker/包络校验不变。

rsync16988 exit0后exec76486三组正确性烟测全部exit0，bitwise_correctness=true。
frontier SHA t1=090b949dfa444ecd319e75492f98def7d90d11f4cde6a66292529a3306f1bf3c，
t2=6bbf9ac4b9ec8f817c31a3020c32e62b66328f113d25d6b9e6c9abcfaa57ea26，
t4=8e1569ba11cf4b541822c3721454ad8b7833f4ef9f396a49278846a717af6cb7，runner5da23966...。
记录三frozen_gather SHA后exec23833启动六场串行正式run.sh，seed613121/2、613141/2、
613161/2。每场533call、5warmup/31formal、NUMA3 CPU240..319、4copy/216MiB scrub/
fixed workspace，geometry及H4096/F512/BF16/SVE256/Ntile16与首块实验一致。
当前采集活跃，尚无多panel泛化结论；1T低M GEMM漂移等未解决问题继续保留。

随后1T第一场validated，rsync62079 exit0后analyze_narrow_prefixes.py 1 1 --history
及evaluate_gather_history.py 1 1均通过。冻结新M评分：shared MAE5.260us/相对MAE5.882%，
packing_paths MAE9.096us/7.564%，两者均无worker超限，第一场均通过预设门槛，但分路径
模型没有在多panel上优于更简单对照。M12控制预测41.813/实测40.09us；packing对
M13/24/48/96预测49.186/83.625/167.250/334.500，实测47.21/77.18/154.22/303.30us。
偏差随完整panel数增加，既有单panel拟合偏差和启动项组合假设都需保留，不能只归为其中一个。
这不是最终六场结论，不据此回调参数；exec23833继续后续场次。

## 多panel六场冻结评分结束，GEMM基线候选建立

exec23833最终六场全部validated/COMPLETE/exit0。依次等rsync12451、35293、53195、82261
exit0后分析剩余场次；analyze_narrow_prefixes --history和evaluate_gather_history均通过，
每场533call、身份/31pair/worker CPU和预测SHA核验保持。无重新拟合或追加样本。
| width | shared相对MAE两场/% | packing相对MAE两场/% | packing绝对MAE两场/us |
|---|---|---|---|
|1|5.882/6.278|7.564/6.442|9.096/8.400|
|2|15.454/15.498|3.536/3.224|2.546/2.118|
|4|18.206/19.349|4.377/5.020|1.334/1.578|
packing六场全部满足相对MAE<=10%和每worker误差<=max(3us,15%)；shared在2T的M17/19
及4T全部新M条件出现worker超限。gather_history_decision.json保存passed_declared_multi_panel_gate。
保留1T shared绝对误差更小和packing随panel数累积偏高，不按宽度挑最佳来重定义模型。

在gather冻结评分结束后，fit_narrow_gemm.py建立独立GEMM候选：首M1..12成本表完全不变，
完整块以固定M12锚点和单一later_full_us拟合M24/36/48/60/96；h1六奇数尾块按同pair相对
M12差取median，映射到偶数kernel row class。尚未测得的偶数尾块/更晚历史只是候选假设。
narrow_gemm.py显式限M1..96/1,2,4T；全部288个候选输出正值，M1..12逐项保持原表，Ruff通过。
第二场同形状重复W13/W2 MAE：1T8.912/16.973us、2T2.356/8.197、4T1.857/3.239；
最大相对误差分别1.304/2.201%、.403/1.801%、.619/1.901%。结果narrow_gemm_candidate.json。
这些重复指标只含本轮M12及以上，不覆盖此前1T/M1/M2状态漂移，不称新M泛化通过。
下一步冻结偶数h1尾块、h2/h4尾块及更大完整块预测，再采独立验证；完整混合宽度provider
仍未接入，16T未知形状和大M范围仍需完成。所有本轮采集/传输/评分已结束，默认不变。

## 窄team偶数尾块／后续历史／大M冻结验证启动

新增narrow_extrapolation.py，保持旧M<=96公式原样，单独标明M97..1718为未验证外推；
未修改被此前冻结预测引用的gather_paths/narrow_gemm。全部5154个(M,width)输出正值有限，
M<=96的288点与旧入口一致，结果extrapolation_checks.json；这不是准确性检查。
prepare_narrow_validation.py生成三组13plan，M12控制和14/18/22、25/35、49/59、
384/697/1200/1718。预测和全部模型/代码/frontier/design SHA在测量前冻结。
M1718在nested sizes中去重，仍保留原anchor；隔离bridge/完整条带和原输入协议不变。

预设GEMM新点每stage条件等权相对MAE<=3%、每点误差<=max(5us,实测5%)；gather单独
使用相对MAE<=10%、每worker max(3us,15%)门槛，M12控制单列，不混合通过率。
evaluate_narrow_validation.py只读冻结预测，分析器增加互斥--validation模式以读取声明
shapes，其余533call/身份/worker/包络校验保持，Ruff通过。

rsync67282 exit0后exec96876三组--correctness-only烟测全部exit0，bitwise=true。
默认extension dd554ea...、calibration cdccff46...不变，runner5da23966...。
frontier SHA：1T2beaf8f4459b8f6cbc7e9fb47a2bbe51db1c42dcb4ef9ea68da8324051ca14de，
2T3645325f2190b97ea56039768665527c7d0260633e686468c2a10cb684e5d946，
4T01167e800c63d5991814678a604d53173dc6ada8613fd93f8f338fb3b87e39c5。
记录三份预测SHA后exec1513启动六场串行run.sh，seed613181/2、613201/2、613221/2；
每场533call、5warmup/31formal、4copy/216MiB scrub/fixed workspace/NUMA3 CPU240..319，
H4096/F512/BF16/SVE256/Ntile16，target96仍在末端1/2/4个CPU，early merge关闭。
当前正式采集活跃，尚无新性能验证结论；不重新拟合参数或扩大已验证的模型范围。

## 窄team新尾块／大M验证完成：W13与gather通过，W2拒绝

exec1513六场全部validated/COMPLETE/exit0；分别等rsync42950、37116、82338、60427
exit0后运行分析器--validation及evaluate_narrow_validation，六场533call/31pair/worker/
来源SHA校验全部通过，M12控制全部通过。未改变参数、门槛或采样数。
| width | gather相对MAE两场/% | W13相对MAE两场/% | W2相对MAE两场/% | W2失败M |
|---|---|---|---|---|
|1|9.428/7.639|.519/.738|2.083/2.141|49，两场重复|
|2|6.663/6.351|.786/.866|2.927/3.005|25/49/697/1718，两场重复|
|4|4.163/4.254|.988/1.004|3.220/3.292|49/697/1200/1718两场重复，384第一场也失败|

gather/W13六场均通过，W2六场均未通过；narrow_validation_decision.json保留分项结果，
status=w2_not_promoted，不将部分通过作为完整provider采用。
第一场W2例：1T/M49预测2579.47、实测2450.98us，误差+128.49us/+5.24%，尽管平均
误差低仍按单点门槛拒绝。2T/M1718预测43377.2/实测41226.1us；4T/M1718为22040.2/
20630.0us，呈系统性高估。不能把h1尾块直接复用到h4或用短范围块成本外推所有大M。

事后描述性w2_late_rate_diagnostic.json：仅取M384与1200完整块差商，1T两场575.510/
575.040us每12行，2T287.610/287.993，4T143.999/143.800；乘width后都在575..576us。
候选短范围常数却为592.769/302.437/153.408us，归一化后约593/605/614us。
这支持区分W2完整块的渐变与后期成本，同时后续尾块历史必须单独处理；它不是物理峰值，
也不单独证明缓存原因。M49实测与另一场次M48的差不能当作同场纯尾块服务。
后续只修W2，保持已通过的W13/gather预测及原失败记录，修后使用新的未拟合点验证。
1T/M1/M2先前状态漂移仍未在本轮验证覆盖。全部本轮任务结束，无活跃句柄，默认不变。

## W2-only暖态修正与新的中间kernel／历史验证

fit_w2_warm.py保留原失败decision后，将旧第一场验证数据标为开发数据。后期完整块
core-normalized成本取三宽度M384/1200差商中位数575.51us；每T固定cold M12锚点，
完整前缀P(h)=h*b+(cold-b)*sum(rho^j)。仅用完整块拟合rho，结果1/2/4T=.732/.824/.813。
尾项保留h1值，后续净变化用(1-tail_rho^(h-1))乘幅度；kernel2/12端点幅度分别拟合，
中间kernel按row class线性插值。tail_rho=.600/.616/.807，kernel2幅度均触及-h1值边界，
kernel12幅度-38.198/-17.851/-16.898us。它是whole-shape净变化，不解释为物理尾块免费。
数学模型同步该候选。W13及M1..12共5154形状检查保持/正值通过，33个h1尾项保持<1e-10us。

w2_warm_candidate.json保存参数、来源和全部旧点评分，第二场开发重复W2相对MAE
1/2/4T=.447/.288/.358%，max2.661/.909/1.102%；这些不挽回原冻结验证失败，也不是新鲜
泛化证据。gather和W13均未重新拟合，1T低M首块漂移仍保留。

prepare_w2_transfer.py固定新M12控制、27/29/31/33(h2中间kernel)、51/53/55/57(h4中间
kernel)、37(h3端点)、192/768(新完整块)，三宽度每组13plan，所有数值预测先行冻结。
沿用此前GEMM/gather分项门槛。分析器和评分器增加显式w2_transfer family，Ruff通过。
rsync36582 exit0后exec15731三组位级烟测全部exit0，bitwise=true，runner5da23966...。
frontier SHA：1Ta0d1bd0b371b3a7e80bdb965a7ab2d0168e570da30bd5e768b6dd73f3093371f，
2Tf7498285ecfda04bc83c87431c569c1362e930dbbc3d15ce63823bfd051c653b，
4T2b9ccac87c410b616e6fbd79735a699c15554274caeebca886a8e26d868db1aa。
记录预测SHA后exec96501启动六场串行run.sh，seed613241/2、613261/2、613281/2；
533call/场、5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3、原完整条带、
H4096/F512/BF16/SVE256/Ntile16保持。当前正式采集活跃，无新点准确性结论或默认变更。

静态检查补充：本轮新增manifest前部条目可独立通过YAML解析；完整manifest在更早的
direct_hybrid_gather等长文本字段中存在未引用的“冒号+空格”，全文件解析失败。
一次只读修复预演发现至少30处既有同类问题，未实际改写这些旧条目，避免扩大本轮改动范围。
此限制已记录，不把全manifest宣称为解析通过；运行脚本和冻结JSON不依赖该解析。

随后1T两场均validated/COMPLETE。等rsync34545 exit0后，用--family w2_transfer分析和
评分两场，533call/31pair/worker/模型来源SHA全部通过。新点W2相对MAE .602/.653%，
无单点超限；W13 .417/.455%、gather8.444/8.284%，控制均通过。此为新1T组合前瞻结果，
不同于旧数据开发重复；不与旧集合MAE直接相减作收益。2T/4T继续原串行采集，未修改预测。

## W2修正版六场新点结束：4T中间kernel仍有残差

exec96501六场validated/COMPLETE/exit0。等rsync83945、17472 exit0后完成2T/4T四场
--family w2_transfer分析与评分，全部533call/31pair/worker/来源SHA校验通过。
W2新点相对MAE：1T .602/.653%、2T .757/.722%、4T1.727/1.504%；gather/W13及M12控制
全部通过。1T/2T W2都通过，4T第一场M31预测427.788us、实测407.18us，高估20.608us/
5.061%；第二场实测408.57us，高估19.218us/4.704%。不修改5%门槛，
w2_warm_transfer_decision.json记not_fully_passed，旧失败版本也仍保留。

差异虽接近门槛，但不只应看这一通过率：4T/M55同kernel8在h4两场高估29.590/28.350us。
以冻结full-prefix模型作描述性扣除，kernel8的h2实际净尾项约88.309/89.699us，h4约
65.748/66.988us；当前统一tail_rho和两端幅度插值预测偏慢。kernel6的h2/h4也有较小
重复残差。这不是唯一物理原因证明；下一步应验证各kernel自身的历史变化，而非为贴边
单点改门槛或添加case常数。已通过的gather/W13和首块成本保持。
所有本轮采集、传输与评分结束，无活跃句柄；完整混合宽度provider与planner未采用。

## W2中间kernel节点候选与新历史验证准备

本轮E类Lab准备及模型语义记录；默认planner、native、Plan V2和原冻结provider不变。
核对现存w2_kernel_history.py/fit_w2_kernel_history.py及candidate来源SHA：中间kernel
4/6/8/10分别使用h1/h2/h4净尾项节点，线性插值并在h4后常数延拓。前次transfer第一场
已用于开发，第二场仅为开发重复；相对MAE1/2/4T=.361/.330/.316%，不是新泛化成绩。
保留旧not_fully_passed，不据此采用新provider；全局目标仍包括16T、混合竞争与完整搜索。

prepare_w2_nodes.py生成新M12控制、39/41/43/45、63/65/67/69、71、199/775，1/2/4T各13plan。
h3检验节点间插值，h5及长历史检验常数延拓，M71保留端点行为对照。所有预测与模型/代码/
frontier/design SHA在采集前冻结；seed613301/2、613321/2、613341/2。沿用原GEMM每stage
相对MAE3%及每点max(5us,5%)、gather相对MAE10%及每worker max(3us,15%)门槛。
分析器和评分器新增显式w2_nodes family，其余评分规则不变。

本轮执行prepare_w2_nodes.py成功；五个相关脚本ruff check通过。独立只读检查验证候选
来源、三份预测SHA、13plan模板等价、门槛不变；5154个形状正值有限、W13不变且h1/首块/
端点保持。预测SHA依次7859c3707af5b674cb39a207a6107d4f162667b47195c5bc4e528bd6f58d8005、
6868c812b4736dbcecf1174c56f535cdf12fff567c9eb6763c18bb3cd54cae5d、
f50774c20462904847f828639eb066be715a06274e3d49091500d4549b560f1a。
当前没有启动远程任务；target bitwise smoke及六场正式验证待执行，无新准确性结论。
完整manifest既有未引用冒号问题仍保留，本轮只验证新增条目，不进行无关格式修复。


随后rsync17704 exit0，exec59256三组正确性烟测均exit0。远程逐项核验bitwise=true、
frontier SHA、runner5da23966...、extension dd554ea...、calibration cdccff46...；三份
frozen_predictions SHA与本地冻结记录一致。frontier SHA1T=f0d5454029a64a58748ad754c2b60b16b1ed91ca9c4c92af7b1020dcc79d12df，
2T=6efdd697782320804d831da941432a9454e69ebd35ba07ad77226db275f41924，
4T=a4d981a7f35804582d6c72d24fd2600dec783376fd4991c06c90766e453d5872。
exec12366启动三目录run.sh的六场串行正式采集，seed613301/2、613321/2、613341/2。
每场533call，5warmup/31formal，4copy/216MiB scrub/fixed workspace；Arm NUMA3 CPU240..319，
H4096/F512/BF16/SVE256/Ntile16，1/2/4T geometry=(T,0,0,1,1)，W13/W2总8/4MiB、
owner8/T及4/T MiB，early merge关闭。当前正式采集活跃，尚无新点准确性结论；默认不变。


## 16T gather旧数据迁移诊断（采集等待期间）

本轮继续轮询exec12366，确认1T第一场正式进程存活，未重启或修改冻结模型。
独立本地audit_gather16_transfer.py直接应用窄team packing_paths六系数到实际16T工作
片段，核对原first_rows/bulk两场isolation/bitwise、31样本及16worker CPU304..319。
它不调用窄team受限预测入口、不修改其支持域，也不拟合16T参数；Ruff通过，输出
width_provider_audit/gather16_transfer_audit.json保存来源SHA和48个shape/session记录。

first_rows条件等权worker MAE两场.654/.723us，相对MAE37.15/42.44%；bulk MAE10.807/
12.038us，相对MAE13.55/13.27%。小M相对误差大但绝对值小；大M方向变成低估。
M1718平均worker预测374.213us、实测worker中位数均值411.431/418.113us；对应stage
包络486.70/489.58us、arrival spread中位数64.11/66.54us。M12 first_rows预测4.215us、
实测3.194/3.111us，而包络58.67/67.13us、arrival54.60/63.95us。各种中位数不可直接相加。
该结果支持后续16T分别验证服务和到达，不能把包络整体拟合成worker带宽或单一倍数。
这是旧数据开发诊断，不是新holdout，尚不提供完整16T模型或planner采用证据。


随后exec12366报告1T/session1 validated并进入第二场。rsync97284 exit0后执行
analyze_narrow_prefixes.py 1 1 --family w2_nodes和evaluate_narrow_validation.py 1 1
--family w2_nodes：533call/31pair/worker/来源SHA全部通过。冻结新点相对MAE为gather
8.334280%、W13 .219073%、W2 .307704%，无单点超限，M12控制通过。仅为第一场1T结果，
不提前宣布六场通过或修改候选；exec12366继续原串行任务。


exec12366随后报告1T/session2 validated/COMPLETE并进入2T。rsync69780 exit0后运行
analyze_narrow_prefixes.py 1 2 --family w2_nodes和evaluate_narrow_validation.py 1 2
--family w2_nodes，身份/533call/31pair/worker/冻结来源全部通过。第二场gather/W13/W2
相对MAE为9.367183/.338609/.471882%，单点和控制全部通过。1T两场W2绝对MAE为
14.369/22.863us，最大单点相对误差.653739/1.063066%。两场均支持该组新历史点，
但2T/4T尚未完成，不做完整provider采用。进程继续原串行采集，不重启、不改参数和门槛。


exec12366继续，2T/session1原始采集及compact分析完成，报告validated后进入第二场。
rsync84497 exit0后运行analyze_narrow_prefixes.py 2 1 --family w2_nodes以及
 evaluate_narrow_validation.py 2 1 --family w2_nodes，全部533call/31pair/worker/来源SHA
核验通过。gather/W13/W2新点相对MAE4.001907/.736692/.629759%，单点和M12控制均通过。
W2绝对MAE19.507455us；最大相对误差M69预测1807.910198/实测1782.870us，高估25.040198us/
1.404488%。保持冻结参数及评分门槛；2T第二场和4T两场仍待完成，默认provider未采用。


exec12366报告2T/session2 validated/COMPLETE并进入4T；rsync98321 exit0后分析器和评分器
2 2 --family w2_nodes均成功，533call/31pair/worker/冻结来源检查通过。第二场gather/
W13/W2相对MAE6.148772/.690728/.587682%，单点及控制全部通过。W2绝对MAE19.393819us，
最大相对误差M63预测1586.440198、实测1563.790us，高估22.650198us/1.448417%。
1T/2T四场全部通过，但4T两场未完成；冻结公式、门槛和默认planner不变，exec12366继续。


exec12366报告4T/session1 validated并进入最后一场。rsync58508 exit0后运行分析器和
评分器4 1 --family w2_nodes，533call/31pair/worker/来源SHA检查通过。gather/W13/W2
相对MAE3.508707/1.386368/.531136%，全部单点和控制通过。W2绝对MAE9.016727us；
h3 M39/41/43/45误差+.391/+1.416/+.971/-.804us，h5 M63/65/67/69误差+7.745/+8.355/
+1.855/-3.535us。长历史M199/775高估32.363/32.606us，对应1.335/.349%，仍保留该偏差。
仅剩4T第二场，尚不写六场最终decision；原冻结模型及门槛保持，exec12366继续。


## W2历史节点六场冻结验证完成

exec12366最终4T/session2 validated/COMPLETE/exit0。rsync83682 exit0后，分析器及评分器
4 2 --family w2_nodes成功，全部533call/31pair/worker/来源SHA通过。第二场4T gather/
W13/W2相对MAE3.461756/1.321581/.528077%，所有单点及控制通过。六场全部通过原门槛，
width_provider_audit/w2_kernel_nodes_decision.json记录passed_declared_kernel_history_gate。
|T|W2相对MAE两场/%|W2绝对MAE两场/us|旧warm公式同点绝对MAE两场/us|
|---|---|---|---|
|1|.307704/.471882|14.369/22.863|16.982/23.903|
|2|.629759/.587682|19.507/19.394|18.644/20.077|
|4|.531136/.528077|9.017/9.018|18.671/18.549|
旧warm比较是在相同测量上的次要事后复算，不是本轮冻结采用门槛。4T改善在两场重复；
1T较小改善，2T一升一降，不能泛称所有宽度明显改进。gather/W13六场及控制全部通过。
保留原warm失败decision和1T低M漂移等边界。当前结论只覆盖同expert独立新历史点，不是
任意历史、16T、混合竞争或planner选择验证。所有本轮采集与传输已结束，无活跃句柄。
下一步补齐16T独立服务及尾块模型；需求和动态竞争、等预算完整搜索仍未完成，默认不变。


## 16T完整块候选建立

本轮E类Lab候选开发，复用影响分析/测试选择/审查流程，仅新增fit_full16.py及记录。
用width16_baseline/analysis.json第一场bulk完整块点，固定M12成本，拟合
P(h)=h*b+(c-b)*(1-rho^h)/(1-rho)。rho网格0..999/1000，每rho闭式最小化相对平方
误差求b，限制0<b<=c。W13(c,b,rho)=93.14/75.450350/.658，W2=44.52/36.449660/.702。
fit_full16.py与ruff check均通过；两个stage h1锚点保持、h1..143正值及单调检查通过。
full16_candidate.json保存数据来源SHA、参数和所有开发/重复点误差；不是原provider采用。

第二场同点开发重复W13 MAE9.276044us/相对MAE.617335%，最大2.059639%；W2 MAE3.440593us/
相对MAE1.018391%，最大4.258065%在M12控制，排除该点后的最大约1.93%。未隐藏M12漂移。
目前仅完整块候选，未拟合尾块/gather、未采新holdout。h140长W2已有线程完成分布异常，
不能直接把负whole-stage净差解释为负尾块成本。下一步应对固定完整前缀的尾项分别检查
短/长历史，再与gather一起冻结16T新点。无远程任务，默认planner及原provider保持。


## 16T尾项相对完整前缀的误差分解

本轮只读分析既有h1/h2/h4/h140两场compact，核验isolation/bitwise和31pair一致，
输出width_provider_audit/tail16_prefix_audit.json共112条stage/history/tail/session记录，
保存来源SHA、固定前缀净差、同场配对whole-stage增量和median-worker服务。未拟合新参数。

h1/h2/h4的完整块相对prefix误差总体小；h140 W13完整块两场相对prefix为-43.89/-39.67us，
因此小尾项相对prefix出现负值（r1约-29.44/-26.04us），不能据此说物理尾块为负。
h140 W2完整块相对prefix为+291.76/+4.38us，r1为-11.19/-10.19us，r11为+24.48/+330.43us；
对应此前线程完成分布诊断，不能用一个尾块历史系数吸收这些包络变化。所有样本保留。

下一步先评估完整前缀的长历史偏差以及尾项的同场配对信息，再形成16T尾块候选；
不能直接将不稳定h140包络差当作冷到热衰减节点。现有full16只是开发候选，未进入原
provider，未启动远程任务；完整混合竞争/搜索目标保持。


## 16T绝对误差前缀及短历史尾项开发候选

fit_full16_absolute.py保持参数量及第一场完整块数据，改用绝对平方误差拟合。
W13(c,b,rho)=93.14/75.009457/.749，W2=44.52/36.289064/.769。第二场完整块W13
MAE9.276->2.718us，M1716误差42.744->.212us，但M60/M96转为+7.247/+9.358us；
W2 MAE3.441->3.891us，没有改善。保存relative版本，统一absolute版本作为开发候选，
不逐点挑参数。脚本ruff和h1锚点/正值/单调检查通过，输出full16_absolute_candidate.json。

fit_tail16.py固定absolute前缀，六种kernel分别从第一场h1/h2/h4取net-to-prefix节点，
线性插值及h4后常数延拓。M1..11原first_rows不变，M12用bulk前缀锚点保持一致。
h140全部保留诊断但不拟合，未筛除异常样本；tail16_candidate.json保存112点评分及来源。
3436个M/stage正值检查及ruff通过。短历史第二场W13/W2 MAE1.915/1.085us，最大相对
3.948/3.471%；长历史两场开发W13 MAE16.674us/最大.397%，W2 MAE47.456us/最大5.638%。
W2长历史异常仍会违反5%单点门槛，不能宣布16T通过或把负净差当物理尾块。
当前只有开发候选，无新holdout和远程采集；下一步补齐gather后冻结16T新点，并保留
长历史完成时间不确定性。默认planner/原provider未变，混合竞争及完整搜索目标未完成。


## 16T gather worker服务候选

fit_gather16.py用既有gather16_transfer_audit逐worker数据和实际work镜像，特征为
[每worker一次启动,片段数,read_KiB,write_KiB]。只用第一场first_rows/bulk，bulk重复M12
不再加入拟合；矩阵rank4，按max(3us,实测)归一化残差做非负最小二乘。
系数[0,.234406782,.332427214,.133418525]，不解释为物理带宽或启动免费。
脚本ruff、M1..1718共27488worker输出正值有限检查通过；gather16_candidate.json保存
全部768worker记录、训练标签、参数、评分及来源。没有拟合到达时间，也没改变原provider。

第二场first_rows绝对MAE.173231us/相对11.030781%，bulk4.071012us/3.499189%；相比旧
窄team迁移bulk12.037831us有所下降。每worker均在max(3us,15%)内，但first_rows相对
MAE仍超过10%，不能把它记为全部门槛通过。当前只有同点开发重复，无前瞻结论。
下一步将GEMM与gather候选一起冻结16T新点，保留长W2包络异常和小M相对误差边界，
不通过临时改门槛声称采用。无远程任务，默认planner不变。


## 16T新点预测冻结

prepare_full16.py用既有16T bulk隔离plan模板创建narrow_full16_t16，anchor保持，
12个独立点为12控制、14/18/22偶数h1尾块、39/43(h3)、65/71(h5)、192/1692完整块、
775/1201大尾块。候选为tail16_candidate及gather16_candidate，采集前冻结全部预测和
来源SHA。seed613361/2、13plans、16T CPU304..319、geometry16/0/0/1/1；其他协议和
原GEMM/gather门槛保持。h140旧异常和M1..11相对误差不是本轮新点通过就自动解决。

分析器和评分器增加full16 family与width16，并拒绝width/family不匹配；旧窄team
families保留。三个相关脚本ruff通过，本地检查13plan除M外模板一致、anchor一致、
16worker预测正值有限、来源SHA和run.sh路径/种子一致。预测SHA为
95783e3f1ac9ef249d29f670356a9d69e6a08827f3aa80cce479156ad866e74f。
当前只完成准备，未启动远程烟测或正式采集；默认provider和planner不变。


随后确认远程无既有本轮目录/采集进程；rsync62379 exit0后exec40424执行16T烟测，
exit0且bitwise=true。frontier SHA7db9496f736be6b87c98f0dcaf832b89490dc1184aeb197291a93876b255897c，
runner5da23966...、extension dd554ea...、calibration cdccff46...均保持；预测SHA与本地
95783e3f...一致。exec87824启动narrow_full16_t16/run.sh两场串行正式采集，seed613361/2。
每场533call、5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，
target16T CPU304..319，H4096/F512/BF16/SVE256/Ntile16，geometry(16,0,0,1,1)，
W13/W2总8/4MiB、owner512/256KiB，early merge关闭。当前采集活跃，尚无新点准确性结论。


exec87824报告16T/session1 validated后进入第二场。rsync20630 exit0后运行分析器/评分器
16 1 --family full16，533call/31pair/16worker/冻结来源检查全部通过。W13/W2相对MAE
.742108/.571288%，绝对MAE7.974441/3.610074us，新点和控制通过。gather相对MAE6.895157%，
但M43/M65单点失败，因此第一场不能整体通过。M43 CPU317/318预测9.865575/9.462765us，
实测5.86/5.89us，高估4.005575/3.572765us；M65 CPU318/319预测12.536597/11.749419us，
实测7.65/7.78us，高估4.886597/3.969419us。保留原max(3us,15%)门槛，不调整系数。
第二场继续exec87824，尚无两场总决策。后续还需解决legacy后续段零请求假设及需求表
real-path transfer未验证边界，独立成本通过本身不构成联合planner可靠性证据。


## 16T两场新点验证结束：gather重复失败与长W2慢线程

exec87824最终session2 validated/COMPLETE/exit0；rsync5545 exit0后分析器及评分器
16 2 --family full16成功，533call/31pair/16worker/来源SHA全部通过。第二场gather/
W13/W2相对MAE7.085989/.674027/1.612478%，控制通过；gather仍在M43/M65失败，W2在
M1692失败。full16_prospective_decision.json保存not_fully_passed，未改门槛或参数。
W13两场绝对MAE7.974/8.074us；W2为3.610/32.919us，不能称整体16T通过。

M43失败CPU317/318，第二场预测9.865575/9.462765us，实测6.11/5.47us；M65 CPU318/319
预测12.536597/11.749419us，实测7.51/7.29us。work镜像显示这些worker主要执行8行尾段：
M43 worker13含64元素12行片段+1312元素8行片段，worker14为1392元素8行片段；
M65 worker14含160元素12行片段+1936元素8行片段，worker15为2160元素8行片段。
这支持下一步检验打包路径系数，尚不唯一证明硬件原因。

W2/M1692固定预测5152.389843us，实测两场5151.45/5461.57us，第二场低估309.180157us/
5.661012%。median_mean_worker为5149.691875/5163.121250us，arrival中位数.58/.47us，
每call max-worker减median-worker的中位数1.85/286.985us。该描述支持慢线程完成分布
变化而非所有worker服务增加309us，不过滤样本、不调高完整块成本吸收。旧h140边界在
新完整块上重现，需要与服务模型分开处理。所有本轮采集结束，无活跃句柄，默认不变。


## 16T gather按打包路径拆分候选

fit_gather16_paths.py保持原标定数据及max(3us,actual)归一化损失，只将需求拆为
[segments8,read8,write8,segments12,read12,write12]六特征，rank6。没有加入旧新点数据
拟合系数；但路径选择使用了旧验证失败证据，因此那组已是开发数据，不再称fresh。
系数为[.618927385,.227991724,.080736985,.291864707,.190083454,.272802019]，有效回归
系数不解释为物理带宽。Ruff和27488worker正值有限检查通过；候选输出gather16_paths_candidate.json。

原第二场first_rows MAE.123736us/8.020220%，bulk4.598751us/3.375526%；bulk绝对MAE
相比统一系数4.071012us退化，保留。所有原重复worker均未超max(3us,15%)。
gather16_paths_development.json保留旧full16两场逐worker复算，排除M12控制后的MAE
2.580509/2.967581us，相对MAE4.073307/4.090241%，M43/M65超限消失且无其他worker超限。
这只是开发解释能力，不是前瞻通过。旧not_fully_passed和长W2慢线程问题仍保留。
下一步冻结未测M的路径比例/边界组合验证；GEMM参数不动，无远程任务，默认不变。


## 16T分路径候选新点准备与烟测

prepare_paths16.py冻结M12控制、26/32/38/44/62/68尾段分配组合、179/180/181/188/193
panel分配切换边界。GEMM候选保持，只替换gather分路径候选。新目录narrow_paths16_t16，
seed613381/2、13plans，原分项门槛不变；分析器/评分器新增paths16 family。
Ruff、模板/anchor一致、预测正值有限和来源SHA检查通过。预测SHA
1986f1ecec8734527cd75490ac819422cf685db686c7dcaf59c34d906b0f5e70。
确认远程无同名目录和残留采集，rsync24433 exit0后exec71037启动位级烟测。
当前尚未正式采集，不把本轮新点通过等同于长W2异常或联合需求问题解决，默认不变。

随后exec71037 exit0，bitwise烟测通过；frontier SHA
aea00331df741df29c23e5df363ed818d07952f5c63e3e27b2b0c1423850e453，runner/extension/calibration
及冻结预测身份全部核验通过。exec66473启动run.sh两场正式采集，seed613381/2，
每场533call/5warmup/31formal、4copy/216MiB scrub/fixed workspace，NUMA3 CPU240..319，
target16T CPU304..319、H4096/F512/BF16/SVE256/Ntile16、完整条带(16,0,0,1,1)，
W13/W2总8/4MiB、owner512/256KiB，early merge关闭。当前采集活跃，尚无新性能结论。


exec66473报告paths16/session1 validated并进入第二场。rsync7773 exit0后分析器/评分器
16 1 --family paths16成功，533call/31pair/worker/来源SHA检查通过。gather相对MAE
4.128967%但M68/M188单点失败：M68 CPU318/319预测11.985655/11.038520us，实测15.10/
15.09us；M188 CPU319预测20.377603us，实测27.60us。低估约3.114/4.051/7.222us，均涉及
完整8行尾段。W13/W2相对MAE1.298542/1.490375%，新点和控制通过。未修改冻结参数，
第二场继续，尚无总决策；路径区分解决旧点不等于所有尾段泛化成立。

等待期间按runner实际copy=pair%4分组旧M1692/1680/1691两场W2：慢线程并不固定在
单一copy，某copy在不同场次的max-worker减median-worker差会变化，不能加入固定copy
惩罚。每组仅7或8个样本，偶数组中位数可位于两模式之间，不作为新的物理模式估计。
此为只读描述性诊断，不筛选样本或改成本；长W2完成分布问题保持未解决。


## 16T分路径两场新点验证结束：完整8行尾段仍失败

exec66473最终session2 validated/COMPLETE/exit0，rsync23055 exit0后分析器/评分器
16 2 --family paths16全部数据校验通过。第二场gather/W13/W2相对MAE3.581675/
1.258974/1.573709%，控制通过；gather失败M44/M68/M188，W13/W2均通过。
paths16_prospective_decision.json保存not_fully_passed。M68 CPU318/319第二场预测
11.985655/11.038520us，实测15.25/15.61us；M188 CPU319预测20.377603us，实测28.91us。
M44 CPU319预测7.334us、实测11.31us。三个M均余8；M68/M188两场重复，M44第二场超限。
完整8行尾段是描述性共同特征，不据此直接加case常数或推断物理原因。

CodeGraph定位gather后直接核对m8函数csrc/moe/arm/common/fused_moe_bf16_tiled.cpp:2720：
8个src指针按有效行置空，循环里分有效源读取与零填充；此处没有显式“valid_rows==8”
不同kernel分支。不能宣称已证明独立全8行kernel切换；实际源流数量、零填充分支、
生成代码及输入地址状态仍可能有关。未修改native或实验参数。所有采集完成，无活跃句柄。
下一步应控制有效源行数量与K片段长度检查该残差，再做新冻结验证；旧两次失败均保留。


## 有效源行6/7/8的配对控制准备

prepare_tail8controls.py创建narrow_tail8controls_t16，四组三点[6,7,8]、[42,43,44]、
[66,67,68]、[186,187,188]，M8作参考控制，seed613401/2、13plans。work镜像逐项确认
每组三点分配完全一致；尾段K长度分别240..304、1312..1392、1936/2160、4096元素。
该实验包含已测点，属于配对机制诊断，不把全组当fresh holdout；不拟合新参数。
GEMM/gather预测仍冻结用于分项记录，M8控制仅用于这轮，旧M12实验门槛与结果不改。

analyze_tail8controls.py预先定义逐pair/worker的7-6、8-7及(8-7)-(7-6)差值，保留31
样本与每个tail worker的work；直接报告配对二阶差，不把各自中位数相减当同一统计量。
增加源行改变读取流，不能直接唯一归因为某指令或缓存。两场重复用于检查可重复性。
四个脚本ruff通过，13plan模板/anchor、来源、种子及工作分配等价检查通过。
冻结预测SHA dafbe2fdfdc018e590370358dcc9af1a0aeb14d837f415eaa871b35efe253882。
当前仅本地准备，无远程任务；配对分析器尚未对新实测数据运行，默认不变。


确认远程无同名目录/采集进程后rsync12638 exit0，exec81873执行配对控制烟测并exit0。
bitwise=true，frontier SHA57dbfca90cb6742b5d02e1cde504b7b02536471c0041b0b1c626d7d34e04443c；
runner5da23966...、extension dd554ea...、calibration cdccff46...及冻结预测dafbe2fd...均核验通过。
exec88198启动narrow_tail8controls_t16/run.sh两场串行采集，seed613401/2，每场533call，
5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，target16T
CPU304..319，H4096/F512/BF16/SVE256/Ntile16、完整条带(16,0,0,1,1)、W13/W2总8/4MiB、
owner512/256KiB、early merge关闭。当前采集活跃，尚无配对结果，不拟合参数。


exec88198报告controls/session1 validated并进入第二场，rsync67484 exit0后执行
analyze_narrow_prefixes.py 16 1 --family tail8controls及analyze_tail8controls.py 1，
533call/31pair/16worker/身份检查通过，输出22个tail-worker配对记录。
第一场base0跨worker的中位D7/D8/D2=.150/.225/.065us；base36 CPU317/318/319的
D7=.91/.82/.81us、D8=3.48/3.75/2.92us、直接配对D2=2.57/2.66/1.78us；
base60 CPU318/319为D7=.26/.55、D8=6.72/6.31、D2=6.51/6.07us；base180 CPU319
D7=1.19、D8=15.46、D2=14.14us。D2直接取配对二阶差中位，不等于两中位之差。
同组work匹配支持第8源行的非线性增量；跨组历史与源地址也变化，不能唯一归因为
K长度或8流硬件上限。第二场尚在运行，参数不动，不提前拟合长度修正。


## 有效源行配对对照两场完成

exec88198最终session2 validated/COMPLETE/exit0；rsync29276 exit0后基础分析器
16 2 --family tail8controls和配对分析器2均成功，533call/31pair/16worker/身份核验通过。
第二场base0跨worker中位D7/D8/D2=.13/.185/.115us；base36 CPU317/318/319的
D2=3.62/3.29/3.63us，base60 CPU318/319为6.40/6.22us，base180 CPU319为13.89us。
对比第一场base60 6.51/6.07us、base180 14.14us，长片段非线性增量重复；base36
第一场1.78..2.66us到第二场3.29..3.63us也有场次变化。所有样本和差异保留。
width_provider_audit/tail8controls_repeat.json保存来源与跨worker摘要，明确诊断完成
而非adoption gate通过。cross-base源地址/历史也变化，不宣称唯一K长度规律或硬件根因。

下一候选限定为全部8行有效片段的服务参数，保留部分有效8行和12行片段公式；用
第一场完整8行服务标定、第二场重复，再冻结未测M验证。不能直接把配对D2机械加到
现有预测（现有7行基线也可能有偏差）。GEMM/长W2问题继续独立处理。所有采集结束，
无活跃句柄，原provider和默认planner不变。


## 完整8行片段服务候选

fit_gather16_valid8.py只替换packed8且valid8片段的服务，保留partial8/12系数及GEMM。
从控制第一场M8/44/68/188的22个tail worker标定，混合worker先减固定12行片段估计，
用max(3us,actual)归一化损失非负拟合a+b*K_elements/1024。矩阵rank2，得到
 a=.056573513us，b=7.298524055us/1024元素。该式为有效服务估计，不宣称硬件带宽根因。
没有将配对二阶差直接加到旧预测；旧候选和两次失败decision均保留。

gather16_valid8_candidate.json保存44个两场worker观测、参数及来源。第一/二场MAE
.193257/.314186us，相对MAE5.255334/8.238088%，均无worker超max(3us,15%)。
这是已测控制点开发重复，不能称fresh泛化。Ruff、27488个worker正值有限检查通过；
M余数非8时逐项与旧分路径模型保持完全一致。未启动远程任务，默认provider不变。
下一步冻结不同历史/worker分配的未测完整8行尾点，保持partial8/12与GEMM作守卫，
继续保留长W2完成分布和联合需求未解决范围。


## 完整8行服务候选新M冻结与烟测

prepare_valid8.py冻结M12控制、20/56/80/104/140/176/200/776完整8行尾点、79/103/108
未改路径guard。新目录narrow_valid8_t16，seed613421/2，原分项门槛；GEMM保持。
分析器/评分器新增valid8 family，Ruff、13plan模板/anchor、正值、来源SHA、GEMM及
非余8路径保持检查通过。冻结预测SHA
ee1e2a6854b5b67de98cefda0035b9a1dc7dc721f54e319334edf673b8ba0e60。
确认远程无同名目录和采集进程，rsync59856 exit0后exec96516启动位级烟测。
当前无新点结论；完整8行候选尚未推广，长W2及联合需求问题不因此缩小或取消。

随后exec96516 exit0且bitwise=true，frontier SHA
c92fbfd6f3599f8f8555948deb3bd4447c0c4ced8458d4e6f1019765c6d8d5d2，runner/extension/calibration
和预测SHA核验全部通过。exec58181启动run.sh两场串行正式采集，seed613421/2；
每场533call、5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，
target16T CPU304..319、H4096/F512/BF16/SVE256/Ntile16、完整条带(16,0,0,1,1)、
W13/W2总8/4MiB、owner512/256KiB，early merge关闭。当前采集活跃，无新评分结论。


exec58181报告valid8/session1 validated并进入第二场，rsync53681 exit0后分析器/评分器
16 1 --family valid8全部数据核验通过。gather相对MAE5.497735%、条件等权worker绝对
MAE1.578150us，但M80/176/200/103单点失败；W13/W2相对MAE1.964024/.988463%，新点和
控制通过。M80尾CPU318/319高估约4.201/8.319us，M176 CPU319高估14.082us；M200 CPU319
预测29.250670us、实测14.42us，高估14.830670us。未改路径M103 CPU319高估3.070817us。
M200与标定M188均完整8行、K4096，时间相差近一倍，证明当前有效行数/K长度候选不足；
跨历史源地址或执行状态仍是未识别维度，不能继续逐M加常数。第二场原参数继续exec58181，
尚无两场总决策，所有失败保留，默认不变。


## valid8新M两场结束：K长度公式未泛化

exec58181最终session2 validated/COMPLETE/exit0；rsync41002 exit0后分析器/评分器
16 2 --family valid8成功，533call/31pair/16worker/来源SHA核验通过。第二场gather/
W13/W2相对MAE5.661190/1.752832/1.344222%，控制通过；gather失败M80/176/200/108，
GEMM两场全部通过。valid8_prospective_decision.json记录not_fully_passed，旧失败保留。
M80 CPU319第二场预测18.188844us、实测10.25us；M176 CPU319预测40.651786us、实测28.06us；
M200 CPU319预测29.250670us、实测15.44us。M108 CPU318预测约25.58us、实测30.28us超限，
该guard路径未改。M200两场14.42/15.44us的较快服务重复，不能使用只由valid8与K决定的
统一慢路径成本覆盖所有输入。

work镜像复核M188 worker15=[(15,15,8,0,4096)]，M200 worker15=[(15,16,8,0,4096)]，
均单一8x4096片段，panel位置及源行集合不同。不能据此唯一归因于输入地址、输出位置、
cache或编译分支；下一步需固定工作量下改变输入位置的对照，暂停继续逐M拟合。
两场全部结束，无活跃句柄。默认planner及原provider未变，完整联合模型目标保持。


## 源token地址类与gather快慢关联

本轮只读核对runner nested_prefixes：保留expert96按token顺序的前M个route，输入
hidden固定2048x4096 BF16，每行8192bytes。直接加载同route文件layer4，快点M8/20/80/
176/200尾8源token奇偶计数均4/4，慢点M44/56/68/104/140/188/776为3/5或5/3。
第二场纯tail例：M80 K2544=10.25us，M104 K3312=24.01us，M188 K4096=28.39us，
M200 K4096=15.44us。混合worker不能直接将总时间解释为尾段服务。

ssh读取cpu319/cache/index0：L1 Data64K、256sets、4ways、64B line；getconf PAGESIZE=4096。
8192B行stride在简单低位索引假设下产生两个set类，5条流超过4way的解释与关联相符，
但实际物理映射/索引方式未验证，4KiB页也不允许直接把更高虚拟位当物理位。
gather16_source_parity_audit.json保存全部两场tail-worker关联、源token、K片段、原数据
SHA及硬件观测。它是事后假设，不增加模型参数或宣布cache根因。
下一步在固定M和输出位置下交换源token选择，比较4/4与3/5，保持计划/工作量；
先证实输入地址关联的干预效果。无远程采集任务，默认不变。


## 固定M的源位置交换干预准备

E类Lab，仅复制现有小runner到source_position_runner，新增source_row_swap.py并在生成
nested route variants后按每plan source_token_row_swaps交换整行top-k。原runner和native
未改，workspace_source SHA保持。新source_position16有anchor+6计划，M188/M200各
base/flip/same；M188交换211<->210使尾源[203..209,211]变为[203..210]、3/5变4/4，
same交换211<->213保持3/5；M200交换225<->226使4/4变5/3，same225<->227保持4/4。
每组固定M、CPU319尾段8x4096、输出panel位置。交换也改变实际源值，不能声称纯物理地址
干预；same-parity对照用于区分一般换行效果。其他expert虽输入位置变化，计数保持且在
隔离target依赖之后执行。

本地对六变体核验每expert count、unique top-k、target M以及前M-8源tokens完全保持；
检查same对照奇偶计数保持，四个非法交换输入拒绝，Ruff通过。design.json记录精确源
tokens、route digest及runner源身份。seed613441/2，7plans，每场预期287calls=35正确性
前缀+7x36。预先分析约定为同pair worker15 gather的flip-base、same-base两场分开，
不拟合参数或筛选样本。run.sh/smoke.sh沿用原NUMA3/默认扩展/4copy/216MiB scrub协议。
当前仅本地准备，未启动远程烟测；原provider、planner和历史实验快照保持。


确认远程无实验目录及进程，rsync32681 exit0后exec32448位级烟测exit0且bitwise=true。
frontier SHA117f4c4e52ffc14c2d4f005ac9d6262c591d8d9b478e82c522c34ca77f7727aa，独立runner
SHA b175f37adcfbae0a9136d7f5010d931fc0847f3ccfaa6f00b27264b160c533a5；extension dd554ea...、
calibration cdccff46...保持。source_position16/analyze.py预先定义逐pair worker15
flip/base和same/base，并校验运行时nested_route_identity tensor SHA与实际声明交换结果，
避免只核对M而未核对输入。初次lint发现import位置，移动到main后Ruff通过；未改runner。
exec77706核验全部runner依赖SHA后启动两场run.sh，seed613441/2，每场287call，
5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，target16T
CPU304..319、H4096/F512/BF16/SVE256/Ntile16、完整条带(16,0,0,1,1)、W13/W2总8/4MiB、
owner512/256KiB、early merge关闭。当前采集活跃，尚无干预结论，默认不变。


exec77706报告source_position/session1 validated并进入第二场，rsync15791 exit0后
source_position16/analyze.py 1成功：287call、31pairs、CPU、runner来源及每变体实际
tensor SHA全部符合声明交换。M188 base/flip/same worker15 gather中位28.10/13.97/
28.04us，配对flip-base=-14.00us、same-base=+.23us；M200为13.54/28.12/13.09us，
配对flip-base=+14.23us、same-base=-.46us。
固定M/输出位置/工作量时改变奇偶分布产生双向约14us变化，同奇偶换行接近不变；
支持源行地址类作为有效特征，比跨M相关性更强，但源值也一起变化且尚未测物理索引，
不能单凭第一场宣称唯一cache根因。第二场原协议继续，无参数拟合或默认变更。


## 源位置干预两场完成

exec77706最终session2 validated/COMPLETE/exit0；rsync1272 exit0后analyze.py 2成功，
全部287call/31pair/CPU/runner及每variant实际tensor SHA核验通过。第二场M188
base/flip/same中位27.89/14.74/27.52us，配对flip-base=-13.03us、same-base=-.30us；
M200为14.71/28.87/14.72us，配对flip-base=+14.02us、same-base=+.38us。
两场改变奇偶占比的干预产生约13..14us双向变化，同奇偶换行对照约-.46..+.38us。
在已测固定M/输出位置/工作量下，源位置选择确实影响服务；支持以实际源token地址类
而非M余8标签选择候选服务曲线。源值一起改变、物理索引未测，不能据此声称唯一cache
机制或任意路由泛化。下一步使用实际source tokens构造条件服务特征，并用新路由/未测
源集合验证；不能假设仅凭expert M就能确定该特征。所有采集结束，无活跃句柄，默认不变。


## 带source-token特征的16T gather候选

fit_gather16_source.py用完整8行片段实际source tokens奇偶最大占数4/5分别估计
 a+b*K_elements/1024；partial8/12和GEMM不改。只拟合旧数据第一场（源位置对照也纳入
开发），第二场均为重复，不称新holdout。参数4类=(1.116484755,3.384607002)，
5类=(.560325324,6.988802631)。第二场tail-worker绝对MAE .406947/1.004375us，
相对10.652137/4.184332%，均无单点超限。4类相对平均仍超过10%，不能隐藏短片段限制。
参数为当前hardware/H4096布局的有效关联，不是物理索引证明。

gather16_source_candidate.json保留所有开发记录与来源。predict必须提供长度M的
唯一非负整数source tokens；完整8行占数>5明确拒绝，不能默认为某条曲线。Ruff、非法
输入拒绝、原route expert96 M1..1718检查通过：1713个M正值有限且非余8与旧模型完全
一致，M464/680/920/1148/1292属于未标定更高占数，被明确拒绝。结果source_checks保存。
因此候选尚不能覆盖所有planner形状；下一步新源集合验证4/5类并采更高占数，不能
通过忽略这些M或沿用任意fallback声称完整目标已实现。无远程任务，默认不变。


## 新源集合及更高占数实验准备

source_occupancy16复用冻结source_position_runner，在M212/M224各构造偶数源数4/5/6/7/8
的完整8行尾段，11plans含anchor，seed613461/2、预期451call。通过整行topk交换保持
全部expert counts、unique top-k、目标前M-8源tokens和固定M工作/输出位置。
M212源集合从230..244范围选择，M224从245..260选择，精确tokens/swaps和tensor SHA
写入design.json。4/5类四个条件使用gather16_source_candidate冻结预测；6/7/8类六个
条件gather预测明确null，仅采集新类别，不能混入预测通过率。
冻结预测SHA5db32fd040cb51c2291257f03a16c5a830511657e0c3856729abf6425172ab1e。

analyze.py预先定义运行时tensor身份、451call/31pair/CPU检查以及四个已标定条件的
原gather门槛，全部worker数据保留；更高占数原样报告服务。初次lint一处单行with
改为正常块后Ruff通过。GEMM预测保存以供独立后续核查，当前评分器仅作gather决策，
不宣称完整provider。当前仅本地准备，远程未启动；无默认模型或planner变更。


确认远程无同名目录/采集进程，rsync29618 exit0后exec2352烟测exit0、bitwise=true。
frontier SHA483315a37a30119eb6f61e6b8de5b7149133a3b0c458ad56181105cb0bcea3d5；runner
b175f37a...及所有依赖、extension dd554ea...、calibration cdccff46...、冻结预测5db32fd0...
全部核验通过。exec70263启动source_occupancy16/run.sh两场串行实验，seed613461/2，
每场451call、5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，
target16T CPU304..319、H4096/F512/BF16/SVE256/Ntile16、完整条带(16,0,0,1,1)、
W13/W2总8/4MiB、owner512/256KiB，early merge关闭。当前采集活跃，无新泛化结论，
已标定四条件与未标定六条件仍分开，不提前增加参数或改变默认。


exec70263报告source_occupancy/session1 validated并进入第二场；rsync37606 exit0后
source_occupancy16/analyze.py 1成功，451call/31pair/CPU/冻结来源及实际tensor身份核验
通过。四个c4/c5冻结条件gather相对MAE3.049358%，单点均通过。
M212 worker15 c4..8实测14.74/29.55/33.10/33.61/35.32us；M224为14.20/28.56/31.84/
33.03/33.82us。c6..8仍只是discovery，没有预先预测，不能计入通过。观察到c4->5大跳变，
更高占数增量较小，不能采用每多一行固定增加14us的线性惩罚。第二场继续原参数，
尚未拟合或宣称高占数泛化；默认不变，联合需求与完整搜索仍待完成。


## 新源集合两场冻结验证完成

exec70263最终session2 validated/COMPLETE/exit0，远程无残留benchmark/analyzer进程。
rsync14539 exit0后source_occupancy16/analyze.py 2成功，451call/31pair/CPU/来源与
实际route tensor身份检查通过。四个c4/c5冻结条件第二场相对MAE2.301686%，无单点
超限；第一场3.049358%。source_occupancy16_decision.json保存passed_declared_c4_c5_gate，
范围仅这四个新源集合条件，不把六个未预测条件计为通过。

第二场M212 c4..8 worker15为14.43/29.57/32.27/33.48/34.44us；M224为14.19/28.70/
31.38/32.49/33.50us，较第一场对应14.74/29.55/33.10/33.61/35.32及14.20/28.56/31.84/
33.03/33.82us接近。高占数增量重复，但全部仅K4096，因此新增高占数候选必须另外
验证K长度变化，不能把同长度重复当所有shape支持。保持c4/c5已冻结曲线再拟合高占数
候选是下一步；同时原partial8/12和短c4相对误差、长W2完成波动、联合需求仍待处理。
所有采集结束，无活跃句柄，默认provider/planner不变。


## 高source占数单参数开发候选

fit_gather16_high_occupancy.py保持c4/c5与其他路径，仅对c6..8采用
G_c(K)=G_5(K)+eta*(c-5)*K/1024。只用source_occupancy第一场六个K4096条件的worker15
相对固定G5残差拟合非负eta，得到.566549731us/1024元素/额外源行。
第二场六条件MAE.960799us；不是新K或新源集合holdout。线性K缩放仅为待验证假设，
不能因为同K重复好就宣称更高占数所有片段可预测。

gather16_high_occupancy_candidate.json保存参数和来源，Ruff通过。原expert96 M1..1718
全部预测正值有限；c<=5及非余8逐项与前版一致，新增候选覆盖原缺失M464/680/920/1148/
1292；构造c6/c7/c8也输出正值。这是数值支持检查，不是完整provider准确性证据。
下一步冻结不同K长度的新源集合验证，未启动远程任务。旧失败、短片段相对误差和长W2
完成时间问题保留，默认planner不变。


## 高占数跨K长度冻结验证准备

source_span16固定M32/92/152，各构造新源占数4/5/6/8，12条件+anchor，seed613481/2，
两场各533calls。原source_position_runner不改；逐变体保持所有expert counts、unique
topk和目标前M-8源tokens。M32尾worker K长度56/1016/1008/1008/1008，M92为1168/2928，
M152为4096，部分worker还包含固定12行片段。全部条件使用high_occupancy候选预测，
无discovery空预测；原gather门槛保持。预测SHA
e49fb1bca25af5994aca2c5e60de68e6ea7447a6f365e3a3fb2e39f064245896。
输入/来源/正值检查及分析器Ruff通过；分析器451->533calls、4->12冻结条件，其余身份
校验和评分规则保持。GEMM预测保留，当前分析器只评gather，不据此作完整provider结论。
本轮仅本地准备，尚未运行烟测或采集，默认不变。


确认远程无同名目录/采集，rsync15794 exit0后exec88334烟测exit0、bitwise=true。
frontier SHA7fc61c7e008a4af27f37441058b82896e71ac8364823ebae855954df343f5acc；runner
b175f37a...及依赖、extension dd554ea...、calibration cdccff46...和预测e49fb1bc...核验通过。
exec50703启动source_span16/run.sh两场串行采集，seed613481/2，每场533calls，
5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，target16T
CPU304..319、H4096/F512/BF16/SVE256/Ntile16、完整条带(16,0,0,1,1)、W13/W2总8/4MiB、
owner512/256KiB，early merge关闭。当前采集活跃，12条件均冻结预测，无新结论或默认变更。


exec50703报告source_span/session1 validated并进入第二场。rsync35785 exit0后
source_span16/analyze.py 1成功，533call/31pair/CPU/来源SHA与实际route tensor身份
核验通过。12冻结条件相对MAE3.521969%，条件等权worker绝对MAE.716214us，单点均通过。
worker15 c4/c5/c6/c8时间：M32=5.68/7.60/7.83/8.53us，M92=13.14/21.52/23.09/23.93us，
M152=21.46/35.26/38.22/41.28us（M152包含固定12行片段，不能当纯尾服务）。
第一场支持高占数增量在所测K片段上的迁移，但第二场尚在运行，不提前记录完整通过或
扩大联合模型范围。原参数/门槛保持，默认不变。


## 高占数跨K两场冻结验证完成

exec50703最终session2 validated/COMPLETE/exit0，rsync37420 exit0后source_span16/analyze.py 2
成功，533call/31pair/CPU/来源及route tensor身份核验通过。12冻结条件第二场相对MAE
3.480976%，条件等权worker绝对MAE.671447us，全部单点通过；第一场3.521969%/.716214us。
source_span16_decision.json记录passed_declared_span_gate，只限本轮source/K组合。
保留带source-token gather候选作后续Lab实验基线，不修改原provider或默认planner；
短片段开发相对误差、partial8/12未覆盖输入、长W2完成分布问题依然存在。
所有本轮采集结束，无活跃句柄。

随后只读核对full_baseline_candidate/baseline.py接口：现有service(model,m,width)只接收M/T，
16T为exact16查表，8T限制M<=1205；没有source tokens参数，1/2/4T不支持。
后续扩展应使用独立Lab接口明确传入源tokens及不确定性，保留旧快照；原median路由
maxM1718还意味着8T大M支持不能凭当前<=1205验证自动扩张。完整接口覆盖和联合需求
仍未完成，不能把本轮gather通过视为整体目标完成。


## 独立Lab成本接口整合

新增width_provider_audit/expanded_baseline.py，E类Lab接口整合及M语义记录；无生产API/
PlanV2/默认改变。Baseline构造时注入原8T service callback，避免复制旧算法或隐式更换
8T模型；加载窄team最终历史候选、窄gather、16T尾项及source条件gather。
service(m,width,source_tokens)返回不可变Estimate：逐worker gather、W13/W2及限制。
16T强制actual source_tokens，输入唯一非负且长度M；8T M>1205显式拒绝，不外推。
限制字段明确隔离/full-stripe范围、未含gather arrival/联合竞争、16T慢线程分布和
1T/M1/M2漂移，不制造未经验证的数值置信区间。旧provider和全部模型文件保持不变。

Ruff通过。用旧baseline.service作为8T callback，逐项比较1/2/4/16T M1..1718及8T
M1..1205共8077组合的gather/W13/W2，完全一致；五类非法/缺失输入拒绝。16T该接线
检查使用顺序tokens，不代表新地址集合精度。expanded_baseline_checks.json保存来源
SHA和检查数量。仅接线验证，不是性能/联合准确性结论；下一步补8T大M覆盖并用真实
route tokens驱动完整计划实验，联合请求和长W2完成波动仍未解决。无远程任务。


## 8T大M外推冻结准备

width_provider_audit/eight_extrapolation.py单独调用原8T GEMM/gather函数，复制原模型
仅将max_m置1718，不改系数或旧provider。M1..1718预测正值有限，M1..1205共1205点与
旧接口逐项一致。large8_extension/candidate.json保留原来源及外推声明，不提前采用。

large8_extension固定expert96/CPU312..319隔离plan（原8T大M标定是expert85），
3个范围内guard M12/1200/1205，9个外推点1212/1220/1283/1344/1405/1487/1572/1691/1718。
seed613501/2、13plans、每场533calls，使用原默认extension和full-stripe协议。
原gather/GEMM门槛保持，evaluate.py只对9外推点评均值，三个guard单列，不混合。
计划经isolate_bridge/validate_resources检查；三脚本Ruff通过。预测SHA
1045bb9983bdb3f120e442f98bfc7d5c068c73ce4c93404435cd1c674398b9e1。
当前仅准备，未启动远程烟测。源expert变更与M范围外推同时存在，guard用于检查
范围内迁移；不能先宣称全部8T范围可靠。expanded_baseline的8T>1205拒绝保持。


确认远程无同名目录/采集进程，rsync9185 exit0后exec82582烟测exit0、bitwise=true。
frontier SHAa84845c84878b130658a50af65953d8936e318ed9b0360d82d377c757ce35b41，runner
5da23966...、extension dd554ea...、calibration cdccff46...和预测1045bb99...核验通过。
exec63545启动large8_extension/run.sh两场串行正式采集，seed613501/2，每场533calls，
5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319，target8T
CPU312..319、H4096/F512/BF16/SVE256/Ntile16、完整条带(8,0,0,1,1)、W13/W2总8/4MiB、
owner1MiB/512KiB，early merge关闭。当前采集活跃，无新准确性结论；旧接口M1205上限保持。


exec63545报告large8_extension/session1 validated并进入第二场；rsync66454 exit0后
analyze.py 1和evaluate.py 1成功，533call/31pair/8worker/来源身份检查通过。9外推点
W13/W2相对MAE .112369/.155339%，绝对MAE20.735259/13.752159us；gather相对MAE
1.657281%，全部单点及三个范围内guard通过。M1200 W13/W2误差+4.329/+6.517us，
M1205 +13.254/+9.253us，M1718 +26.155/+19.417us，当前未见明显跨expert基线偏移。
这是第一场结果，第二场继续原参数exec63545；旧接口上限不提前改变，默认不变。


## 8T外推两场通过与显式Lab接入

exec63545最终session2 validated/COMPLETE/exit0，rsync89303 exit0后analyze.py 2及
 evaluate.py 2成功，533call/31pair/CPU/来源核验通过。第二场9外推点gather/W13/W2
相对MAE1.190785/.146479/.162306%，全部单点和三个guard通过；W13/W2绝对MAE
27.125259/14.185252us。large8_extension_decision.json记录passed_declared_extension_gate，
仅此选定M/协议/expert范围，不声称所有route或联合环境可靠。全部采集结束，无活跃句柄。

expanded_baseline.Baseline新增显式extended8 callback，仅在存在通过的冻结decision时
允许配置；M<=1205继续旧legacy8，1206..1718才调用扩展。未提供callback仍拒绝大M，
不静默切换；旧provider和模型文件保持。1718个8T输出与冻结外推公式完全一致，
恰好513次调用扩展，未配置扩展拒绝检查和Ruff通过，expanded8_checks.json保存结果。
本轮为Lab接口接线，生产默认无变更。下一步可以使用完整宽度候选与actual source tokens
做联合需求/竞争实验；长W2完成波动及未验证地址响应仍须保留，不能据此完成整个目标。


## 完整历史计划的独立成本替换对照启动

E类Lab诊断replay_expanded.py读取真实route layer4并为224个expert生成actual source tokens，
确认总12288routes；使用expanded Baseline及显式8T扩展回调构建anchor/control_forward。
保留旧provider对照，均使用原31个独立arrival scenarios、setup/gap/publication、
shared capacity、small8响应和local16 W2响应。只替换隔离service并按相同legacy分段
重建first/later；需求字节及后续零请求假设不改，以区分基线变化与竞争项变化。
与已有两场完整计算完成时间比较，不拟合新参数，不称新holdout或搜索效果。
脚本Ruff通过，exec35327本地回放启动，输出expanded_joint_replay，尚无最终结果。
默认planner、生产及旧实验快照不变；长W2完成波动/联合需求缺口继续保留。


exec35327完成四组31场景回放exit0。old anchor29088.960409us复现前次capacity结果，
expanded29068.013268us，减少20.947141us；两场实测28697.71/28642.22us，误差从
1.363/1.560%变1.290/1.487%。old control_forward29832.187793us也复现前次结果，
expanded29822.079646us，减少10.108146us；实测29650.79/29776.00us，误差从.612/.189%
变.578/.155%。两组都是8T/16T，不能支持1/2/4T联合泛化或新搜索质量结论。

expanded_joint_replay/summary.json保留全部结果；identity_checks核验31场景数、同一
coefficients、每job的放置/依赖/gather请求以及每段除service_us外的所有字段均完全一致，
固定需求/响应对照成立。只有独立成本变化带来的推进/重叠反馈改变，不拟合任何竞争系数。
新成本仅小幅改善这两组历史完整时间；下一步重点是窄team联合及后续请求假设，而非
继续用这些历史结果回调独立基线。当前无运行任务，默认不变，整体目标仍未完成。


## 窄team联合持续背景诊断准备

prepare_narrow_joint.py固定前台expert96/M48，1/2/4T分别位于CPU319、318..319、316..319。
背景固定expert85/14/115/59，实际M1205/768/714/529，全部8T；local在280..311，cross
在248..279。每width local/cross n0/1/2/4加anchor共9plans，seed613521/2、613541/2、
613561/2，每场369calls。完整输入通过nested_prefix保留top-k/active experts，核验四
背景M不变；cohort_bridge和validate_resources确认依赖/资源。长背景用于覆盖前台后续
阶段，而非只测首块burst。未启动硬件实验，原模型/默认不变。

freeze_narrow_joint.py使用expanded独立成本、实际tokens、31个独立arrival场景和同一
capacity/response参数；仍保留legacy首段8/4MiB请求、后续零请求/零敏感度。每条件只
模拟初始active cohort（其余任务在整个cohort完成之后），保存原阶段时间及相对同locality
n0的delta。预设每stage delta MAE<=5us、max<=12us，诊断失败也保留不回调参数。
四背景预测W13 delta：1T local/cross22.838/22.826us，2T26.636/26.518us，4T15.173/
15.190us；W2均数值零。这是待检验模型预测，不是实测结论。

准备脚本Ruff通过；冻结脚本初次两处分号lint失败，在采集前仅拆为三行，刷新来源身份，
所有预测对象逐项保持，最终Ruff及全部冻结source SHA检查通过。输出narrow_joint_t1/t2/t4
各自frozen_predictions.json；未测量、未拟合联合响应。下步完成评分器/烟测后正式采集。


新增evaluate_narrow_joint.py，预定义同pair前台W13/W2相对对应locality_n0阶段差，
保留31样本、绝对阶段时间和背景GEMM重叠比例。重叠取所有实际背景W13/W2区间并集，
不重复计重叠时间；三个空/重叠/分离区间检查及Ruff通过。评分沿用每stage六个非零背景
条件delta MAE5us/max12us。冻结来源、369call/种子/CPU/阶段包络身份均检查；该重叠
仅证明同时运行，不直接证明物理流量。原模型参数和冻结预测未改。
确认远程无同名目录和残留任务，rsync78120 exit0后exec11502启动三组串行位级烟测。
当前烟测活跃，尚未开始正式采集，无新的联合性能结论。

exec11502三组烟测全部exit0；bitwise=true，runner5da23966...、extension dd554ea...、
calibration cdccff46...和三份冻结预测SHA核验通过。frontier SHA1T035416a69352d432661caa4317e387de5d955dcdc7210df1a45e1837c8ea1be3，
2T3d6a89e7c4825efab7d80c40792ff64b7b6ced3f1af1840f9ef3e5a8db8d0442，
4T4c6ed02993bc4e94ec8afa92901f7a52d79aa96503b79ad12dcce52ba74dde23。
exec43797启动三目录run.sh六场串行采集，seed613521/2、613541/2、613561/2，各369calls，
5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319、H4096/F512/
BF16/SVE256/Ntile16、early merge关闭；全部full-stripe，前台owner W13/W2=8/T和4/T MiB，
背景8T owner1MiB/512KiB。当前采集活跃，尚无联合减速结论，无模型参数或默认变更。


exec43797报告narrow_joint_t1/session1 validated并进入第二场。rsync62246 exit0后
 evaluate_narrow_joint.py 1 1成功，369call/31pair/CPU/阶段包络/冻结来源检查通过。
W13 delta MAE19.601416us/max30.096205us，W2 MAE10.498333us/max20.33us，两stage按原
5/12us门槛失败，但不能描述为漏掉大幅持续减速。
local n1/n2/n4实际W13 delta=-8.52/-4.18/+2.96us，W2=-12.27/-20.33/+5.40us；
cross对应W13=-4.91/-5.03/-7.27us，W2=+.11/+5.94/+18.94us。W13无背景约4.86ms，
W2约2.45ms，变化幅度小，负值全部保留。W13背景GEMM重叠比例约.912..978，W2均1.0。
当前W13预测+8.5..22.8us主要偏高；有重叠不等于有显著共享资源竞争，此结果不支持
将后续零请求直接认定为当前主要漏项。2T/4T与重复场次仍待完成，exec43797继续，
不调整门槛或参数，不从第一场作普遍结论，默认不变。


exec43797报告1T/session2 validated/COMPLETE并进入2T。rsync94899 exit0后
 evaluate_narrow_joint.py 1 2成功，全部身份/369call/31pair/worker检查通过。
第二场W13 delta MAE13.736416us/max25.136205us，W2 MAE14.626667us/max25.09us，按原
门槛两stage仍失败。local n1/n2/n4实际W13=.06/4.02/8.26us，W2=-23.42/-20.65/-25.09us；
cross W13=-3.72/1.93/-2.31us，W2=-3.37/-10.32/-4.91us。W2背景GEMM重叠均1.0。
两场没有稳定大幅减速，负delta不筛除；1T数据不足以支持新增显著持续竞争惩罚。
同样不能据此证明后续请求物理为零或所有背景配置无竞争；当前只测试长8T背景。
2T/4T继续原串行任务exec43797，参数/门槛和默认保持。

随后exec43797报告2T/session1 validated；rsync92899 exit0后评分2 1成功。W13 delta
MAE6.487779us/max14.268363us，W2 MAE3.63us/max16.60us，按原门槛均未通过。
local n1/n2/n4 W13 delta4.38/11.60/20.39us，W2 .04/-2.24/.65us；cross W13
4.87/9.71/12.25us，W2 .65/-1.60/-16.60us。无背景W13约2433.5..2433.95us、W2
1241.4..1241.99us；W2背景重叠均1.0。小幅W13减速出现，仍无显著持续W2减速证据。
2T第二场及4T两场继续，不回调参数或改变门槛。


exec43797报告2T/session2 validated/COMPLETE并进入4T。rsync44097 exit0后
 evaluate_narrow_joint.py 2 2成功，身份/369call/31pair/worker检查通过。W13 delta
MAE6.049445us/max13.548363us未通过，W2 MAE4.135us/max9.57us通过原门槛。
local n1/n2/n4 W13 delta6.29/10.72/23.46us，W2=-4.48/-2.23/-6.64us；cross W13
7.61/4.78/12.97us，W2=.38/-1.51/-9.57us。W2背景重叠均1.0。与第一场相比，W13小幅
减速方向接近，W2仍无明显持续减速；不能把一个stage一次通过当完整联合模型通过。
4T两场继续原串行采集exec43797，参数/门槛和默认不变。


## 窄前台＋四个长8T背景六场结束

4T/session1在rsync46267 exit0后评分成功：W13 delta MAE2.295657us/max9.577357us，
W2 MAE3.966667us/max9.31us，均通过。local n4 W13/W2 delta24.75/-5.66us，cross
15.02/.82us，W2实际重叠比例1.0。
exec43797最终4T/session2 validated/COMPLETE/exit0；rsync44935 exit0后评分4 2成功，
W13 MAE3.977761us/max14.407357us未通过，W2 MAE1.39us/max2.60us通过。
第二场local n4 W13/W2=29.58/-2.60us，cross=15.34/-1.39us。六场全部数据身份、369call、
31pairs、CPU和冻结来源检查通过；narrow_joint_decision.json保留not_fully_passed。

结果只支持本组背景下的小幅W13竞争：4T local较cross额外约10..14us重复，W2没有
显著持续减速，即使GEMM全程重叠。未满足原严格门槛的场次全部保留，不回调系数使其
通过，也不据此认定物理后续流量为零。当前背景始终只有四个8T team，不能代表
1T*32、2T*16等更多竞争者共享LLC的情况。
下一步固定背景32核预算，比较不同窄team数量/宽度，并基于nested后实际且未被donor
改写的expert counts选择背景；不能直接使用原route counts选入被重分配的expert。
所有本轮采集结束，无活跃句柄，默认不变；联合模型和完整搜索尚未完成。


## 固定32背景核的窄team组合准备

prepare_core_budget_joint.py构造4T/M48前台与32x1T、16x2T、8x4T、4x8T背景，
local/cross各四组合加各自bg0及anchor共11plans，seed613581/2，每场451calls。
background pool为nested后计数未变的32个最大expert，M1205..73；排除被重分配的0/1
及target96。全部expert count/unique top-k由既有nested机制保持，模型使用actual形状。
固定背景核数不等于固定expert集合/计算工作量，本轮是不同组合的预测验证，不是纯width
因果ablation。generic cohort bridge保留剩余DAG并等待active cohort结束，资源检查通过。

freeze_core_budget_joint.py固定expanded服务、31arrival场景及全部旧capacity/response，
保留后续零请求。local背景1/2/4/8T的W13 delta16.239/15.940/15.802/15.173us，W2
22.670/3.256/1.728/0us；cross W13约15.19..16.25us，W2约0..22.63us，位置效应仍很小。
core_budget_joint/evaluate.py定义同pair相对locality_bg0的8条件stage delta及实际重叠，
门槛仍MAE5us/max12us。三个脚本Ruff、全部冻结source SHA、前台4核/背景32核检查通过。
当前预测与输入准备完成，未启动远程烟测；默认和旧实验快照不变。


确认远程无同名目录/采集，rsync96634 exit0后exec42451烟测exit0、bitwise=true。
frontier SHA32a5aae7b4cbd1d24f44ad7642cd41202a891e51a95bc1bb7066b33fb54ab206；runner
5da23966...、extension dd554ea...、calibration cdccff46...及预测3464d726...核验通过。
exec4556启动core_budget_joint/run.sh两场串行正式采集，seed613581/2，每场451calls，
5warmup/31formal、4copy/216MiB scrub/fixed workspace、NUMA3 CPU240..319、H4096/F512/
BF16/SVE256/Ntile16、early merge关闭；全部full-stripe，前台4T owner2MiB/1MiB，
背景T1/2/4/8 owner8/T和4/T MiB。当前采集活跃，无新竞争结论或模型参数变更。


exec4556报告core_budget/session1 validated并进入第二场。rsync11537 exit0后evaluate.py 1
成功，451call/31pair/CPU/阶段包络/冻结来源检查通过。W13 delta MAE21.572879us/
max46.660805us，W2 MAE10.956353us/max34.894295us，均未通过原门槛。
local背景1/2/4/8T实际W13 delta62.90/55.86/50.83/30.49us，W2=42.81/38.15/12.33/2.26us；
cross W13=28.26/26.50/28.27/14.55us，W2=16.47/10.70/4.51/-3.29us。W13无背景约
1224us、W2约627..629us，W2实际背景重叠均1.0。local32x1T的W13/W2约5.1/6.8%增量，
明显高于四8T背景；local16x2T W2模型3.256us而实际38.15us，局部压力响应缺口更明确。
背景expert集合也随组合变化且已输入模型，不能将全部差异唯一归因于宽度或物理LLC；
但同组合local/cross提供位置对照。第二场继续原参数exec4556，未拟合任务数惩罚或
提前宣布联合修正，默认不变。


## 固定背景核两场结束

exec4556最终session2 validated/COMPLETE/exit0，远程无benchmark/analyzer残留；
rsync66257 exit0后evaluate.py 2成功，451call/31pair/worker/冻结来源检查通过。
第二场W13 delta MAE13.314129us/max28.250190us，W2 MAE7.687603us/max25.574295us，
均未通过；core_budget_joint_decision.json记录not_fully_passed。local背景1/2/4/8T
W13 delta42.03/44.19/35.70/22.52us，W2=32.02/28.83/10.54/.22us；cross W13
24.59/23.29/22.73/12.56us，W2=20.25/10.39/6.38/-3.30us。局部减速方向重复，但local
幅度较第一场小，场次差异全部保留，不用单场拟合任务数惩罚。

下一步需要窄team独立需求测量，以将后台需求与前台敏感度分开。只读核对已有
l2_stage_demand8/measure.py与protocol：只覆盖8T，M12/48/1205，1/32copies、synthetic
contiguous routes，64DDRC+24core事件，whole-stage budgets且非真实4copy/scrub路径。
large8_stage_demand native源kWidth=8硬编码。已有测量不可直接当1/2/4T需求，扩展需独立
Lab构建及正确性/PMU扰动检查，不能把L2事件直接转换为精确LLC bytes。
本轮未修改native或启动新采集；所有当前任务结束，默认planner不变。


## 窄team独立请求测量准备

本轮E类Lab native测量工具扩展，复用影响分析/测试选择/审查流程并读取C++相关规则。
新增narrow_stage_demand/stage_service.cpp，来自旧独立8T harness，唯一宽度配置为
编译期FUSED_CPP_LAB_STAGE_WIDTH（仅此standalone build，static_assert1/2/4/8），
允许M12/48/96及idle；原JIT kernel、生产扩展和全局build不改。保留每cell常数W13输出
和W2全输出/sentinel检查、affinity/SVE256、team barrier、64次lead-in和completion-edge计数。

t1/t2/t4分别有独立measure/analyze/protocol/run脚本和旧PMU依赖快照，core事件范围
随width变化（CPU319、318..319、316..319）；DDRC仍64事件，另加3*T core事件。
M12窗口100ms、M48/M96窗口500ms；1/32copies、W13/W2分开、5warmup/31formal、
无PMU控制加两场PMU，seed613601/2、613621/2、613641/2。quality_limits预设one-count
fraction<=5%、PMU/control偏差<=5%、counter running>=99%，未达条件不能直接用于拟合。
计数窗口和lead-in有成本，正式轮次耗时应在烟测后按实测估算；当前不启动采集。

Python修改文件Ruff通过；native尚未编译，不能宣称数值或性能通过。build.sh仅产出
独立t1/t2/t4及8T对照binary；protocol native_sha待构建后核对更新。whole-stage合成
需求不能替代真实4copy/scrub route状态或直接解释为精确LLC bytes。下一步目标构建、
常数输出/guard及PMU烟测，再核对扰动与窗口误差。当前无运行任务，默认不变。


## 窄team请求工具构建、烟测通过并启动正式采集

本地/远程jit_kernels.cpp SHA1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629，
header c92aaf60...一致。rsync14423 exit0，exec29035独立1/2/4/8T构建exit0，未改生产
build/extension。t1/t2/t4 native SHA分别77617c897a774da8afedc3ddc2c62d63b0fb107373755000a5ffc4348f904e45、
380ebb538facae42e6e6df1804d603f224fa002d1b7b0ed7e2bd871778de9f05、
52de409eaed15c2c70dd2e6ac1cb06e28deafbf9f0f9e9847d7e79da94eef84d，写入protocol并回传。
新增analysis quality标记count_bound_ok/pmu_control_ok/eligible_for_request_fit，超限记录
不删除、不直接用于拟合；Ruff通过，rsync69555同步仅分析脚本。

exec80407三宽度无PMU各13cell烟测通过，exec93505三宽度PMU各13cell烟测通过，均核验
grid/常数数值/输出sentinel/CPU和counter覆盖；core事件分别3/6/12加64DDRC。8T对照
独立执行M12/48/96两个stage共6cell，numerical/CPU通过，非性能比较。rsync98856 exit0
取回smoke原始JSONL、protocol和build identity。
smoke_review.json：最大one-count fraction1/2/4T=.020/.009901/.004975；基于64lead-in+
实测window估计三正式runs各13.62/10.86/9.49分钟，不含allocation/输出/控制开销，
整轮预计35..45分钟。单轮烟测不证明PMU扰动已合格，正式无PMU对照仍必须执行。

exec29289已启动三宽度串行run.sh（每宽度control+session1/2），随后各session分析。
沿用NUMA3 CPU240..319/controller240、worker末端1/2/4核、SVE256/BF16/H4096/F512、
M12/48/96、1/32copies、synthetic contiguous routes、64lead-in、5warmup/31formal、
100/500/500ms窗口。此独立需求协议没有真实4copy/scrub连续expert状态，不能直接替代
runtime T0。当前正式采集活跃，尚无需求/扰动结论，默认模型和planner保持。


## 请求采集期间的配对位置响应目标

继续轮询exec29289，远程核验t1/control的measure.py及native存活，观察到round8/9；
未重启任务。新增placement_response_targets.py对已有narrow_joint和core_budget数据
逐pair计算(local-bg0)-(cross-bg0)，52个condition/stage/session记录保留全部31样本，
Ruff通过。P10/P90是经验样本分位，不是置信区间；cross不是纯DRAM，位置差也非LLC唯一原因。

固定核预算32x1T的W13位置额外中位为36.17/19.37us，W2=20.21/15.95us；16x2T为
W13=29.74/17.97us、W2=25.05/14.44us。直接paired差的中位数不等于两个中位数之差。
第二场32x1T W13 P10/P90=-5.54/50.12us、W2=-9.63/34.58us；16x2T也跨零，保留
波动，不把中位数当无误差精确服务代价。较宽背景的额外影响更小且噪声宽。
输出placement_response_targets.json供后续需求/敏感度分层开发；不修改竞争参数。
独立请求正式采集仍在运行，尚无完整PMU/control质量结论，默认不变。


请求采集exec29289继续。远程核验1T/control已到round31并结束，现t1/session1的
measure.py/native存活；未重启。rsync87643 exit0后本地t1/analyze.py control成功，
完整grid、31rounds/5warmup、常数数值、CPU、binary SHA和completion计数检查通过。
control最大one-count fraction2%，全部count_bound_ok；无PMU所以不作请求拟合资格。
1/32copies服务近似：M12 W13=1233.68/1229.74us，W2=606.89/606.11us；M48 W13=
4936.59/4931.37us，W2=2430.64/2431.69us；M96 W13=9875.17/9862.86us，W2=4860.00/
4867.70us。服务近似不等于资源请求相同，等待PMU正式数据。

本地request_feature_geometry.json检查旧8T/16T第一场DRAM读写总量率与L2 refill事件率
两列设计，列范数归一化后rank均2，条件数8T W13/W2=3.143/5.780，16T=3.862/7.884。
这只是旧数据数值设计审查，不拟合响应，不证明物理分离或窄team迁移；请求rate与L2
事件保持各自单位，不转换成精确LLC bytes。当前无竞争系数改动，正式采集仍活跃。


## 1T第一场PMU通过与持续请求增量

新增narrow_stage_demand/increments.py，按同round idle-subtracted per-call预算计算
(M48-M12)/3和(M96-M48)/4；输出DRAM读写MiB与L2 refill/writeback事件的直接配对差，
保留负值、P10/P90及端点quality flags，不称单独计时的panel流量。Ruff及单位归一化/
有符号扣除检查通过。
exec29289报告1T session1 complete，远程核验session2进程活跃。rsync9827 exit0后
本地t1/analyze.py session1及increments.py 1 1通过；12个正式cell全部eligible_for_request_fit，
最大count边界2%，PMU/control服务偏差绝对值约<=.105%。这是测量质量，不是模型泛化。

每增加12行的L2 refill事件持续存在：W13约124821..138745，W2约61202..66794。
W13 DRAM读MiB增量（12->48 / 48->96）为1copy1.1511/1.0919，32copy3.6205/1.4786；
W2为1copy.1327/.2111，32copy1.7097/1.2997。whole-stage时间1/32copy近似但请求不同，
不能从服务时间推断请求相同，也不能把所有L2 refill当DRAM或将后续请求设为物理零。
L2事件未换算精确LLC bytes，配对差非直接panel测量。第二场与2/4T继续原采集exec29289，
不从一场建立通用竞争参数，默认不变。


## 1T两场PMU完成：事件增量较稳，DRAM状态变化保留

exec29289报告1T两场及分析完成，进入2T。rsync92704 exit0后本地increments.py 1 2
成功。第二场12cell全部eligible，最大PMU/control服务偏差.153240%，计数边界仍<=2%。
测量质量通过不等于请求率重复性通过，increment_repeat.json保留32个feature/条件差。

W13每增加12行L2 refill第一/二场：1copy12->48=124821/129472、48->96=129576/135931；
32copy=132263/133460和138745/139758。W2为1copy62960/62981、61202/61106，32copy
65497/65386、66794/68154。事件增量相对稳定。
DRAM读增量MiB第一/二场：W13 1copy1.151/1.390及1.092/1.400，32copy3.620/3.738及
1.479/1.984；W2 1copy.133/.659及.211/.465，32copy1.710/1.706及1.300/.869。
尤其W2单份权重DRAM变化大，不直接平均成稳定常数，也不因仪器质量通过而忽略。
可优先将L2事件作为候选压力特征，DRAM需保留状态/重复不确定性；尚未拟合竞争参数。
2T/4T沿原协议继续exec29289，默认不变。


## 精确测量格的请求bank

继续核验exec29289，远程2T/control measure.py/native存活，观察到round25/26。
新增narrow_stage_demand/build_request_bank.py，仅消费两场完整analysis和同binary协议。
生成t1/request_bank.json共12个M/stage/copies精确格，全部仪器质量合格；每格保留
两场request vector、standalone service、计数边界与扰动，另给逐特征repeat min/max。
保持MiB/call和events/call，不裁剪负值、不平均、不换算LLC bytes；repeat范围不是置信区间。
显式temporal_segments=null、real_path_transfer_validated=false、interpolation_supported=false，
缺失宽度拒绝，不借其他T。脚本Ruff和12格/两session/native身份检查通过；未拟合竞争系数。
正式2T/4T采集继续原任务，无新运行或默认变更。

## 2T两场请求采集完成并建立独立需求表

重新核验exec29289报告width2 measured and analyzed；远程ps确认原串行任务及4T/control
measure.py/native正在运行，未重启采集。rsync39949 exit0后，使用既有t2/analyze.py control、
increments.py 2 1、increments.py 2 2、build_request_bank.py 2全部exit0。
2T原始control/session1/session2 JSONL及两场分析已保存在narrow_stage_demand/t2。
沿用上述NUMA3、CPU318..319、BF16/SVE256、H4096/F512、Ntile16、full stripes
(2,0,0,1,1)，完整W13/W2权重8/4MiB、owner4/2MiB；5warmup/31formal、1/32copies。

两场各12cell全部仪器质量合格，最大one-count fraction .009901；相对无PMU控制的
最大服务偏差分别 .110510%/.124203%。request_bank.json保留两场独立向量，不合并为
确定常数，尚无时间片段或真实route迁移验证。

每增加12行的paired whole-stage请求差中位数（第一/二场）：

|stage|copies|M区间|DRAM读MiB/added12|L2 refill events/added12|
|---|---:|---|---|---|
|W13|1|12→48|.8221/.8679|119306/117818|
|W13|1|48→96|1.1454/1.1112|120759/121867|
|W13|32|12→48|3.6081/3.5248|130918/129078|
|W13|32|48→96|1.2393/1.2812|138117/136179|
|W2|1|12→48|.0838/.1903|46530/46321|
|W2|1|48→96|.0798/.1974|45390/44705|
|W2|32|12→48|1.7965/1.7855|64376/63267|
|W2|32|48→96|.8155/.8399|63166/64128|

2T也存在持续的后续请求；warm W2的DRAM小信号跨场变化明显，L2事件较稳定。
这些差值不是直接测得的单panel请求，也不是LLC字节。当前证据支持分别保留局部事件
与DRAM需求，不支持把首块后需求置零或将所有局部事件送入DRAM容量约束。
未更改模型参数、planner默认或生产代码。4T仍沿原任务采集，完整联合模型验收未完成。

## 固定核预算联合数据的实际阶段覆盖审查

读取core_budget_joint/frozen_predictions.json及两场session*_compact.json，按真实前台
W13/W2时间窗积分每个背景expert的gather/W13/W2重叠，保存width_provider_audit/
core_budget_stage_overlap.json。共32个condition/stage/session记录、992个样本窗口，
每样本背景活跃核积分不超过32；保存源身份和逐pair结果。未推断panel边界，活跃核数
仅描述阶段覆盖，不作为新压力模型特征。沿用core_budget_joint已记录的4T/M48前台、
32背景核、两场5warmup/31formal、真实route与full-stripe协议。

发现前台W13期间背景gather不能忽略：第一场local_bg1/2/4/8的时间平均活跃gather核
中位分别16.850/16.097/14.523/10.327；前台W2期间分别4.781/3.902/.477/0。
第二场对应16.837/16.089/14.644/10.185及4.734/3.908/.432/0，cross相似。
各阶段分别取中位数不可直接相加解释同一个样本。
全部992窗口的背景W2重叠为零。因此这组响应不能单独辨识背景W2竞争系数；即使
W13/W2需求表完整，也不能将该数据全部减速拟合为GEMM竞争。必须保留gather需求项，
并在后续真实路径错峰条件中补充背景W2覆盖。

窄需求表M12/48/96与此组背景实际M没有精确匹配（背景73..1205）；当前bank明确
禁止隐式插值，不能直接接表声称联合验证。下一步需显式定义且独立验证需求随M/进度
的组合规则，或以已测M构造匹配背景对照；这不改变完整真实计划验收目标。
4T/control原进程仍存活，session1.stdout尚不存在仅表示尚未进入PMU第一场，非任务失败。
本轮未修改竞争系数或生产/default，已得到影响拟合设计的实际阶段覆盖证据。
后续ps核验：4T/control已完成31轮，已进入session1 PMU，measure/native PID607001/607002
存活；继续原exec29289，无重启。

## 1T/2T需求随M线性组合的结构审查

使用已生成的t1/t2/request_bank.json，固定第一场M12/M48，计算
Q96_pred=Q12+(96-12)/(48-12)*(Q48-Q12)，逐stage/copies/资源与两场实际M96比较。
完整64条记录存narrow_stage_demand/linear_demand_composition_audit.json；此为已有数据
回顾性结构审查，非新鲜留出，也未修改需求表或竞争模型。测量协议沿用上述窄team记录。

8个width/stage/copies条件的L2 refill误差范围：同场-2.714%..+1.523%，跨场
-5.531%..+2.795%。DRAM读同场-18.958%..+40.649%，跨场-50.573%..+40.855%。
32copies的同场DRAM误差，1T W13/W2分别+35.654%/+11.962%，2T分别+40.649%/+32.394%。
说明简单恒定后续DRAM增量本身已有明显误差，不仅是跨场噪声；不能沿用到M1205背景。
局部事件的近线性只是候选结构证据，尚未通过新M、真实路径或竞争环境迁移。
DRAM需明确过渡/历史分段；只有M12/48/96三点还不能判定96以后斜率稳定，也不应
为了拟合这三个点增加没有留出证据的自由参数。
exec29289仍存活；远程PID607001/607002确认4T/session1 PMU在round15/16运行。

## 4T第一场完成：宽度与复用状态共同影响请求

exec29289报告session1 complete，远程ps确认session2 measure/native PID609745/609746
存活。rsync99247 exit0取回完整control/session1；本地既有t4/analyze.py control、
t4/analyze.py session1及increments.py 4 1均exit0。第一场12cell全部eligible，最大
PMU/control服务偏差.111315%，最大one-count fraction.004975。第二场尚未完成，
不生成要求两场的request_bank，不提前宣称重复性通过。
协议为NUMA3 CPU316..319、BF16/SVE256/H4096/F512/Ntile16、full stripes(4,0,0,1,1)，
W13/W2全权重8/4MiB、owner2/1MiB，其余同原窄team需求协议5warmup/31formal。

第一场M48的whole-stage L2 refill events/call：

|stage|copies|1T|2T|4T|
|---|---:|---:|---:|---:|
|W13|1|492091|461992|373638|
|W13|32|526759|523493|501346|
|W2|1|241523|177738|42629|
|W2|32|261842|258639|249064|

同M下服务时间近似随宽度缩短，但请求量并非统一缩放：尤其warm W2的4T refill显著
少于1T/2T，32copies时差异较小。宽度与复用状态需共同进入需求表达，不能借用窄/宽
team统一请求表；该跨宽度比较不是同时运行的带宽或缓存机制因果实验。
4T每增加12行的L2 refill配对增量，W13 1copy为99884/111562，32copy为123295/146796；
W2 1copy为12504/10069，32copy为61291/59212（12→48 / 48→96）。DRAM读增量MiB
依次为.2145/.2431、3.3059/.8365、.0324/.0152、1.8032/.8882。
原始有符号读写增量及全部31样本保留，局部事件不转换为LLC bytes。未改竞争参数。

## 复用条件与宽度交互的全格审查

narrow_stage_demand/reuse_width_interaction_session1.json保存三宽度、三个M、两个stage
的18组32copy/1copy比较，来源各t*/analysis_session1.json。4T/W2在M12/48/96的
L2 refill比率为13.053/5.843/5.876，而对应耗时比率1.0077/1.0029/1.0036。
需求变化远大于无竞争耗时变化，再次说明不能以T0代替资源需求；这些是独立条件
描述性比较，不是共同运行的服务压力，也未验证cache具体因果机制。

CodeGraph查询未定位ignored tmp准备器，定向读取prepare_core_budget_joint.py及
prepare_narrow_joint.py。现有cohort_bridge通过整expert task_deps组织前台和背景，
head均无依赖，没有GEMM阶段级启动条件。后续背景W2错峰不能仅重排现有head声称
达成；需要可复现的启动方式并核验实际trace覆盖。尚未修改runner或计划。

## 窄team需求采集完整结束与跨场审查

exec29289终止exit0：session2 complete、COMPLETE、width4 measured and analyzed。
远程ps不再有t4 measure/native；未重启。rsync28353 exit0回传t4/session2.jsonl和
analysis_session2.json；本地increments.py 4 2、build_request_bank.py 4均exit0。
三宽度control+两PMU全部完成，t1/t2/t4/request_bank.json各12精确cell均两场仪器质量
合格，保留原始记录、不同会话及有符号请求，未合并为确定常数。4T第二场最大PMU/control
服务扰动.113467%。此批原始协议、native身份、CPU/window/geometry沿用以上记录。

新增narrow_stage_demand/all_width_repeat_audit.json共144条feature/cell比较，用
200*(Q2-Q1)/(|Q2|+|Q1|)描述对称相对差，不选择任一场作真值，不称置信区间。
L2 refill最大绝对对称差：1T3.823%、2T2.531%、4T12.591%。4T最大点是M96/W2/1copy，
82687.591→93798.178 events/call；因此不能概括所有局部事件都高度稳定。
DRAM读最大绝对对称差：1T120.891%（M48/W2/1copy .5146→2.0874MiB）、
2T64.932%（M96/W2/1copy .8066→1.5820MiB）、
4T46.093%（M96/W2/1copy .1840→.2942MiB）。保留小信号绝对量以免百分比误导。
仪器质量合格不等于需求重复性或真实联合迁移通过。

此批完成后的实际下一步是将需求状态和重复范围作为显式输入，修正联合模型后续请求
为零的假设，并以分阶段实测时间线检查响应；gather贡献、背景W2覆盖及未测M需求
组合仍须处理。当前没有活跃采集，生产/default和竞争系数未变，完整目标仍未完成。

## 独立需求接口实现（M类Lab，未接入默认模拟器）

使用impact-analysis/test-selector/code-review-gate，范围为新增narrow_stage_demand/
request_curve.py及test_request_curve.py；消费既有bank，不修改T0、生产API、runtime或
planner默认。MATHEMATICAL_MODEL.md同步公式，manifest新增model.narrow_request_curve，
回退边界为此独立模块。

RequestCurve默认精确查询M12/48/96，显式piecewise在相邻节点插值，显式late_linear
以48..96斜率延伸至1718；两者对未测M均返回unvalidated标记。M<12拒绝，无小M需求
数据时不伪造。每场/每宽度/1或32copies独立，未插值复用状态；原始负write值保留。
Demand.average_rates(D)仅返回预算/D的平均率，不能自动解释为阶段内恒定压力。
这使预算与T0独立且减速后平均率相应降低；时间分配与响应接入仍待完成。

.venv/bin/pytest -q tmp/joint_cost_model_20260911/narrow_stage_demand/test_request_curve.py
结果19passed；两源Ruff检查通过。request_curve_checks.json核验三bank的72条观测
逐字段精确复现，并记录120条显式候选组合，均有限。这些检查只证明接口和单位，
不证明插值/外推、尾块、真实route或完整计划泛化；没有新增性能采集或准确率声明。
审查保留whole-stage与temporal需求边界，未把有符号噪声直接当非负服务容量请求。

## 需求接入实测时间线的第一版条件响应诊断

延续M类Lab及impact-analysis/test-selector/code-review-gate范围。新增
narrow_stage_demand/fit_trace_response.py和test_trace_response.py，不改runtime/default。
固定core_budget_joint的前台T0，从实测阶段时间线积分背景平均请求率；四特征为
共享DRAM读、同域L2 refill、共享/同域gather逻辑率。显式假设whole-stage均匀请求、
未测M用late_linear；不称为逐panel流量。gather仍是逻辑工作量，未计独立写压力。
1/32copies与需求场1/2独立计算，主探索情景32copies/需求场1，不挑最佳情景。

.venv/bin/pytest -q tmp/joint_cost_model_20260911/narrow_stage_demand/test_trace_response.py
四测试通过，覆盖重叠边界、负delta不裁剪、单位缩放和非负边界解；两文件Ruff通过。
.venv/bin/python tmp/joint_cost_model_20260911/narrow_stage_demand/fit_trace_response.py
输出trace_response_diagnostic.json，包含逐pair特征、四情景、第一场拟合、第二场
重复及背景宽度留出。读取已完成且isolation/bitwise/451calls校验通过的两compact。

主情景第二场六条件W13/W2 delta MAE10.108/5.211us，旧模型同条件16.089/9.663us。
输入利用实测前台窗口及背景耗时，有目标时间信息；不能作为与自主旧模型的公平泛化
改进。主W13所有第二场条件高估，local_bg1/2/4误差+21.01/+13.10/+15.16us。
W13拟合共享DRAM系数为零，gather特征主导，不能物理解读为DRAM不产生竞争。
W2留1/2/4T背景的第二场MAE10.654/4.696/10.839us；对应旧5.862/16.396/6.732us，
跨宽度不稳定。四参数六训练条件、留宽度仅四条件，W2留4T归一化condition353.15。
不能通过增加系数消化这些误差；当前数据不足以稳定分离gather与GEMM贡献。

结论为未通过响应门槛，不接入planner默认。下一步需要独立gather竞争对照或不同阶段
覆盖来分离响应，并验证请求的时间分布；完整计划、自主时间线和等预算搜索仍待完成。
MATHEMATICAL_MODEL公式及manifest已同步，回退为独立诊断模块，无新增远程实验。

## 已有独立gather对GEMM影响的反向角色证据

读取gather_with_gemm/measure.py和analyze.py后确认，原实验不仅记录gather服务，也记录
同域8T/M12/W13 GEMM的每team调用数，且包含同peers/copies的无gather控制。
复用analyze.load验证control/session1/session2完整grid、31formal、数值/CPU/计数，
以每round的n*1e6/summed_bg_calls_s计算等效调用周期，并与无gather同round配对。
输出narrow_stage_demand/gather_to_gemm_evidence.json保留21条件记录、全部逐round值
与源身份。未重新采集，无新拟合。等效周期包含同步/调用余项，不等于计时GEMM服务。

独立8T gather位于312..319，GEMM位于此前8*n核，同NUMA3/LLC；BF16/SVE256、
H4096/F512、Ntile16、full stripes(8,0,0,1,1)，W13完整8MiB/owner1MiB。
gather M529或1205、输入/输出32copy，GEMM1/2/4team及1/32copy，100ms窗口，
原5warmup/31formal协议。双方保持工作到ACK，native窗口并非完全同一时间戳。

M1205 gather对32copy GEMM的配对周期增量，session1/2：
1team2.82/2.82us，2team3.32/3.94us，4team6.18/6.71us。
4team/1copy则.83/.98us；无PMU控制对应4team/32copy6.20us、1copy.68us。
4team/32copy的周期中位session1为178.43→184.70us，session2为178.43→185.13us；
配对差中位不等于中位之差。其他条件和负差样本均保留。

此独立证据支持gather确实可改变GEMM吞吐，且影响依赖GEMM复用状态；不能仅由
联合trace回归中gather系数较大就判断它是伪特征。但宽度8T/M12/W13、稳态吞吐
与当前4T/M48/W13/W2阶段包络不同，不能直接冻结这些数值为后者系数。
下一项匹配对照应固定4T/M48 W13或W2，与独立gather压力同时运行并保留双方吞吐；
这样才能从联合回归中独立约束gather项。当前无远程任务，默认与模型参数未改。

## 匹配4T/M48的独立gather竞争探针

E类Lab，使用既有impact-analysis/test-selector/code-review-gate及C++并发/平台规则。
新目录gather_against_narrow复用小型stage_service.cpp，仅将GEMM pin和CPU核验端点
320改312，4T使用308..311；gather复用已验证gather_io_states/native_matched在312..319。
不改kernel、生产build或runtime。coordinator派生gather_with_gemm测量协议，记录双方
服务/调用数、64DDRC事件，双方持续运行直到ACK。15cell含idle、独立gather、独立
W13/W2及联合条件；GEMM M48、1/32copies，gather M529/1205、IO32/32copies。
NUMA3同LLC、BF16/SVE256/H4096/F512/Ntile16，GEMM(4,0,0,1,1)，stage8/4MiB，
owner2/1MiB；500ms窗口。CPU不同于真实trace316..319，使用本探针相同核位solo对照。

rsync69751 exit0；exec36098构建+两烟测exit0。remote JIT SHA1bcdaf58...与原一致，
gather binary SHA f47b8336...固定；新native完整身份见build_identity.txt和protocol.json。
rsync89011 exit0回传两完整JSONL；analyze.py smoke_control/smoke_pmu各15cell通过，
含常数数值/guard、CPU、grid、counter running>=99%、onecount<=5%，后补binary检查
重新load两文件通过。measure/analyze Ruff通过。单轮不作为性能/扰动验收。
正式计划control+session1/2，5warmup31formal，seed613661/2，预计15..20min；
无新增资源范围，沿既有Arm授权。分析器区分GEMM median服务与等效cycle，保留配对差。
正式PMU/control扰动、重复性及真实route迁移尚待验证，模型系数/default不变。
rsync43151 exit0后exec49304启动原定run.sh，正式control及两场PMU串行；当前运行中。

## 匹配gather正式采集期间的比较流程

核验exec49304仍运行，remote PID617945 run.sh、618010 measure.py、618036 GEMM
native存活；control处于预热阶段，未重启。新增gather_against_narrow/compare.py及
两项test_compare.py测试：按round键配对、负delta保留、缺失round拒绝，2passed，
两文件Ruff通过。使用已完整烟测数据跑通12个GEMM cell、8个joint contrast检查，
不依据单轮烟测给正式性能结论。

比较器要求control及两场全部完整，formal31/5与预定seed匹配；分别比较GEMM median
service、GEMM等效cycle、gather cycle，并保留逐round差、median和MAD。MAD为样本
散布，不是置信区间。PMU/control各指标相对差5%及count5%标记独立保存；跨场差异
可能包含环境漂移，不能全部解释为PMU扰动。当前未生成formal comparison.json。
仅新增本地分析流程，未修改正在运行的measure/native、T0或响应参数。

## 配对响应质量标记补齐两侧对照

B类本地分析修正，沿用impact/test-selector/review范围。发现compare.py的联合质量
标记只查joint PMU cell，遗漏被相减的solo GEMM、solo gather，以及无PMU控制侧
的计数质量。三个新增回归用例在修改前均失败：solo count超界、control count超界、
solo PMU/control扰动超界仍被joint标合格。
修正为joint和两个solo reference均检查PMU/control指标及双方count上界，保存全部
reference_pmu_control_pct而非仅总布尔值。原始配对差/median/MAD不变，不裁剪负值。
同一pytest命令现5passed，Ruff通过；两烟测12GEMM cell的reference字段检查通过。
当前尚无formal comparison输出，故没有覆盖旧结论或重算已发布性能。
exec49304仍运行；remote PID618010/618036存活，control已观察到round10/11。
未改远程measure/native或运行顺序，仍等待原任务完成。

## 独立gather与联合trace的逻辑压力单位核对

读取gather_io_states/gather_service.cpp确认每次8T调用执行完整hybrid gather，各worker
处理其工作片段；freeze_core_budget_joint.py使用joint_gather_work.worker_features的
read_bytes+write_bytes构成gather_kib。用同一helper核验冻结计划全部130个job实例，
每worker逻辑量完全一致；实际source code的计时循环未修改。
新gather_against_narrow/logical_feature_identity.json保存探针M529/1205逐worker特征，
输入读取量均等于M*4096*2，合计read+packed-write分别8.3203125/18.8515625MiB。
后续独立响应特征用Qlogical/gather_cycle_us，单位MiB/us，与trace诊断口径一致；
不得除以worker时间之和或再次乘8。这是逻辑量身份核对，不证明物理DRAM请求相同。
原exec49304仍运行，PID618010/618036确认control已到round22/23，无新采集。

## 匹配gather无PMU正式对照完成

exec49304报告validated15cells31rounds count bounds True；ps确认已切换session1，
measure/native PID624567/624569存活。rsync78055 exit0回传control.jsonl和analysis_control。
本地load再次核验31/5、seed613661、grid/native身份；control_response.json保存完整
配对服务与cycle差。该文件自比较的PMU字段仅是恒等值，不作扰动结论。maxonecount
.002519，通过5%边界。沿用本探针4T/M48、NUMA3同LLC、gather8T M529/1205协议。

|gather M|GEMM|copies|solo/joint service中位us|paired service delta us|MAD us|paired cycle delta us|
|---|---|---:|---|---:|---:|---:|
|529|W13|1|1234.94/1236.71|1.71|.29|3.111|
|529|W13|32|1242.01/1248.80|6.81|.27|6.295|
|529|W2|1|611.33/610.70|-.64|.19|2.302|
|529|W2|32|613.94/618.11|4.22|.23|6.240|
|1205|W13|1|1234.94/1237.02|2.14|.31|3.111|
|1205|W13|32|1242.01/1250.20|8.13|.24|9.455|
|1205|W2|1|611.33/610.59|-.74|.13|3.069|
|1205|W2|32|613.94/619.02|5.08|.14|7.027|

单场已有复用状态响应差异；warm W2服务delta为负而cycle为正，不能将吞吐变化全部
写入kernel service成本。MAD是场内散布，不是重复会话验证，等待两场PMU。

进一步冻结前一轮conditional fit的shared+local gather系数，对本对照32copies条件
计算T0*Qlogical/实测gather_cycle*k，未重拟合：W13 M529/1205预测29.263/31.672us，
实测6.81/8.13us；W2预测55.339/60.001us，实测4.22/5.08us。
输出conditional_coefficient_transfer_control.json。协议不同，不能唯一归因于物理机制，
但明确不支持将原joint回归的gather系数作为可迁移常数。其吸收其他遗漏项的可能性
需独立约束后检验；不凭本单场重新拟合或改变生产/default。

## 独立gather约束后的响应求解器准备

延续M类Lab流程，新增narrow_stage_demand/constrained_response.py及8项定向测试。
在原四特征结构上固定kg_shared+kg_local=k_independent，所有系数非负；消去局部
gather系数后对shared系数加0..k上界。复用已测nonnegative_fit，比较内部解和上界解，
保留带符号y及降维矩阵rank/condition。此约束只适用于明确同域总响应的线性候选，
不把独立实验误称为能区分shared/local；总量须由调用方输入，不能从joint y暗中重拟合。

.venv/bin/pytest -q tmp/joint_cost_model_20260911/narrow_stage_demand/test_constrained_response.py
8passed，覆盖上下边界、可手算精确解、仅local暴露不可辨识及非法总量拒绝；Ruff通过。
MATHEMATICAL_MODEL同步公式。尚未生成任何真实数据系数，未使用单场对照作正式校准。
exec49304仍在运行；PID624567/624569核验session1到round9/10。采集程序与默认模型不变。

## 独立gather压力覆盖与真实trace的范围差异

用完整无PMU control的逐round调用周期和已核验逻辑量计算probe工作率，并与
trace_response_diagnostic.json中同一口径的X_gshared/T0比较。保存
gather_against_narrow/pressure_coverage_control.json。
32copy GEMM条件probe中位逻辑率：W13 M529/.02873、M1205/.03110MiB/us；
W2分别.02875/.03118。M增加主要同时增加工作量和周期，压力幅度仅有小范围变化。
真实第一场同/跨域、1/2/4T背景的前台W13窗口平均gather率为.04721..06113MiB/us，
各条件31样本均高于相应probe样本范围；前台W2为.00073..01708，各31样本均低于probe。
两场逐条件的below/above计数保存在artifact中，不是瞬时peak或物理DRAM流量。

因此本批独立数据只能约束一个有限压力区间的响应，固定gather系数仍含幅度外推，
不能将只换gather M的两点当作广压力校准。后续需增加压力档位，且真实窗口中部分
时间无gather的脉冲重叠不一定等价于稳态低率，需单独验证。现有gather binary明确
只支持teams=1，不能通过修改命令中的team数就启动多team压力；不改变正在运行协议。
exec49304仍活跃，PID624567/624569确认第一场PMU已到round22/23。

## 匹配gather第一场PMU完成及调用计数分辨率

exec49304报告session1 complete并继续；rsync4890 exit0后本地load再次验证31/5、
seed613661、native身份、完整grid和counter。comparison_session1.json保留全部参考
质量字段：12/12 GEMM cell合格，三类参考指标最大PMU/control差.290462%。第二场未完成。

paired service delta us（control→session1）：
M529 W13 1copy1.71→1.66、32copy6.81→6.84；W2 1copy-.64→.46、32copy4.22→4.38。
M1205 W13 1copy2.14→2.00、32copy8.13→8.12；W2 1copy-.74→-.51、32copy5.08→4.93。
32copy响应接近；warm W2小量存在跨零，不能将质量通过当成正减速已识别。

cycle_count_resolution.json进一步按每role计数n的[n-1,n+1]观察相位区间换算cycle，
再形成joint-solo差范围。第一PMU W13每一个计数对应约3.10..3.17us周期变化，
W2约.77..79us。M529/W13/1copy的31样本有16个差区间包含0；其余多数不包含0。
因此整数计数能解释部分cycle台阶，不能把所有cycle-service差异归因于它，特别是W2。
该区间只在稳态计数相位假设下成立，不是统计置信区间或非稳态误差界，也不用于
native median服务计时。优先保留服务与cycle两个结果，不将二者混合拟合。
当前未生成正式重复comparison或新响应系数，继续原第二场采集。

## 第一PMU独立系数冻结与固定gather的联合诊断

M类Lab新增gather_against_narrow/freeze_response.py，复用8项已通过约束求解测试，
新脚本Ruff通过。仅load完整control/session1并检查全部引用质量，第二PMU未读取。
每stage/复用条件用两gather M的median x=T0*Qlogical/实测gather_cycle、paired y
做零截距非负标量拟合。independent_response_session1.json保存源身份、逐round特征、
无约束/有约束系数和训练误差；负y不裁剪，仅参数有非负约束。
W13 1copy/.048226131、32copy/.201719062；W2 1copy/0、32copy/.253004140。
W2零边界不证明无竞争效果，候选仅在窄压力段与合成路径，真实4copy状态不等价。

随后只对已读过的core_budget_joint历史数据做约束诊断，冻结独立kg总量并重拟合其它
项；背景需求情景固定32copy/需求第一场，前台1/32copy假设分别保留，不择优。
constrained_joint_diagnostic.json保存两情景、stage和留背景宽度结果。
主32copy前台假设第二旧会话MAE W13/W2=10.515/4.713us，max18.507/16.266us，
仍未通过。W13 local1/2/4背景高估18.507/11.777/17.568us；W2 local4高估16.266us，
local1/2仅1.528/.945us。W2留1/2/4T背景MAE3.479/3.732/15.732us，最后max29.114us。
固定独立gather后仍存在宽度迁移问题，需求时序与前台敏感度仍需验证。

该诊断继续使用实测时间线和未验证的长M需求外推，不是自主预测/新鲜计划验收；不
据此选择复用假设或修改planner默认。MATHEMATICAL_MODEL/manifest已同步。
原exec49304第二PMU仍运行，PID631257/631259确认round7/8；冻结未受第二场结果影响。

## 固定gather后的剩余项与需求情景敏感性

gather_against_narrow/constrained_terms.json对既有冻结系数做代数分解，非物理时间归因。
W2第二旧会话local_bg4预测DRAM10.200us、local_refill16.380us、gather.226us，实测
总delta10.54us；local_bg1为18.171/12.714/2.663us，实测32.02；local_bg2为
14.131/13.480/2.164us，实测28.83。模型局部请求项在4T背景过大，而gather已经很小。
不能据此称实测有16us LLC stall，这是拟合项的诊断分解。

进一步固定同一个前台32copy独立gather系数，分别使用后台1/32copy及需求场1/2，
每种用旧联合第一场拟合、第二场检查。constrained_demand_scenarios.json保留全部
8个stage/scenario，不按分数选状态。W2 local_bg4误差：1copy需求场1/2为+10.95/+9.95us，
32copy为+16.27/+16.29us；对应六条件MAE5.518/5.259/4.713/4.623us。
W13六条件MAE10.849/10.471/10.515/10.264us，同域4T仍高估。
单纯换复用端点或需求场不能消除该宽度迁移问题；请求时序、实际资源服务路径和
前台敏感度仍需独立区分。未新增权重惩罚，不把最优情景当真实state。
exec49304仍运行，PID631257/631259确认独立第二PMU到round23/24。

## 匹配gather全批完成及冻结系数重复验证

exec49304终止exit0，输出session2 complete/COMPLETE；rsync39497 exit0回传完整
session2 JSONL/analysis。运行compare.py，正式comparison.json生成，两场各12GEMM
cell（含引用对照）全部仪器质量合格。原任务结束，没有新增或重启实验。

使用第二场读取前已冻结的independent_response_session1.json，按同一实测gather周期
和独立GEMM服务口径计算预测，保存independent_response_repeat.json全部逐round结果。
8条件配对误差中位的绝对均值.286928us，最大.790us；这是同条件重复性，不是新压力
形状或自主模拟验收。单份W2零参数的第二场实际-.17/-.79us，继续保留负差，不称正
减速已被识别。
32copy第二场W13 M529/1205预测7.179/7.803us、实际6.54/7.97us；W2预测4.465/4.850us、
实际4.35/4.98us。独立响应的窄区间重复性较好，但已知压力覆盖不全、steady/真实
路径不同、实测周期条件化限制保持。联合4T背景的局部项高估未因此解决。
下一步围绕GEMM竞争的实际请求时间分布/资源路径与前台敏感度做区分，不再放大gather
吸收残差；完整动态预测、新鲜计划和等预算搜索仍未完成。

## 总量不变的前缀请求时序对照（未解决4T高估）

M类Lab新增request_timing.py，给定实测stage起止，用m/M进度映射M12/48/96及最终M
节点，将独立累计请求差分配到各段。总量与原query一致，负read/refill增量拒绝，
原始write有符号保留；非instrumented panel，不能称实际冷/热阶段。fit_trace_response.py
新增--timing prefix，仅跑预定32copy/需求场1，uniform默认和原输出文件保持。
原T0/gather特征/paired y逐记录一致检查通过，记录于prefix_timing_comparison.json。

3项守恒/区间积分/减速反馈测试及既有4响应测试共7passed，三修改源Ruff通过。
.venv/bin/python tmp/joint_cost_model_20260911/narrow_stage_demand/fit_trace_response.py --timing prefix
生成trace_response_prefix_diagnostic.json。使用固定独立gather总量约束进一步比较：
只改时序、保留原竞争系数，第二旧会话MAE W13/W2为46.476/21.921us；
另行重拟合其余响应后10.251/4.543us，同域4T的W13/W2仍高估17.562/16.238us。
uniform对应10.515/4.713us与17.568/16.266us。均未通过，前缀时序并未消除关键残差。
不能用重拟合后的接近误差宣称时序已正确；两种结构可由参数部分补偿，参数辨识仍不足。

MATHEMATICAL_MODEL与manifest同步。无新增远程任务，生产/default不变。
下一步直接GEMM对GEMM匹配对照，固定前台M48/4T，独立改变背景宽度/复用状态，
以区分请求代理量与前台敏感度；不继续增加宽度惩罚或采用未验证时序假设。

## 直接GEMM对GEMM匹配探针实现与烟测

E类Lab新gemm_pair_response/measure.py、events.py、analyze.py，复用既有已核验binary，
未改kernel或重新构建。前台4T/M48位于308..311（gather_against_narrow/native），
背景1/2/4T/M48位于最后1/2/4核（narrow_stage_demand/t*/native），同LLC且无核重叠。
NUMA3/BF16/SVE256/H4096/F512/Ntile16，双方full stripes(t,0,0,1,1)，stage8/4MiB，
每owner8/t与4/tMiB。W13/W2、1/32copies独立组合，每width25cell（idle+4soloFG+
4soloBG+16joint）。200ms窗口，双方持续至ACK，64DDRC加3*(4+width)核心事件，分别
保留前后台L2 rate；native窗口与PMU窗口不完全相同，不把core事件换成精确LLC字节。

rsync92069 exit0，exec67223六轮smoke完成exit0；rsync32398 exit0回传原始JSONL。
各width无PMU及PMU一轮25cell均数值/CPU/grid/binary/事件覆盖通过，counter running>=99%。
最大onecount1/2/4T为2.5%/1.25%/.633%；六分析文件及smoke_review.json已保存。
measure/events/analyze Ruff通过。无PMU smoke耗时11.76/9.41/8.26秒，PMU
11.81/9.46/8.31秒；按三width、control+2PMU、5warmup31formal投影53.15分钟，
包含重复启动开销的保守线性估计，非正式性能结论。

按既有Arm实验授权准备run.sh串行整批，输出不覆盖，全部identity在protocol.json。
单background team用于先分离width/stage/reuse，不能代替真实32背景核或完整planner
验证。正式PMU扰动/重复性未完成，无模型系数或default修改。
rsync71434 exit0后exec74373已启动run.sh，三width各control/session1/session2串行。

## GEMM配对的请求量／吞吐分离分析准备

原exec74373仍存活，remote PID642051 run.sh、642116 t1_control measure.py确认
预热完成进入正式轮次；未重启。新增gemm_pair_response/decompose.py与3项单元测试，
沿用E类Lab分析流程，不改正在运行measure/analyze/native。
每role按同round idle扣除事件率，以Qsolo=(Rsolo-Ridle)/lambda_solo、
Qjoint=(Rjoint-Ridle)/lambda_joint分开每调用预算与调用率。固定预算预期联合率
为Qsolo*lambda_joint，残差=(Qjoint-Qsolo)*lambda_joint；保留负预算及事件残差，
不将其直接称为LLC字节或唯一cache机制。

3项测试覆盖仅减速预算不变、额外请求和负idle-subtraction保留，pytest3passed、
Ruff通过。三width完整PMU烟测各32个role contrast通过有限值/代数一致检查；不将
一轮烟测当正式请求变化结论。分析器仍区分service与cycle、双方独立事件与全局DDRC，
native/PMU窗口差和计数分辨率未消失；正式PMU/control质量及跨场重复仍待完成。

## 双GEMM正式汇总与两侧质量对齐

新增gemm_pair_response/summarize.py消费各width完整control和两PMU，要求31/5及
预定seeds，调用既有load/decompose。每joint的foreground/background角色分别输出
service delta、cycle delta、service MAD、同round请求预算与事件率残差，保留所有
原始配对值；分量median不要求可加，不当作物理stall分解。
质量检查覆盖joint双角色、soloFG、soloBG的service/cycle PMU差异以及PMU/control
双方onecount边界，每joint8个扰动项。Ruff通过，三width完整烟测各16个joint质量
组装检查通过；未用单轮烟测生成正式系数或repeat结论。
原exec74373继续，remote PID642116确认t1_control到round10/11，无重复启动。
当前未生成正式summary_t*.json，不改变采集源码、模型参数或planner默认。

## GEMM配对1T背景无PMU对照完成

exec74373报告t1_control775条正式cell记录、maxcount .025、elapsed422.292s。
rsync12097 exit0回传完整t1_control.jsonl与analysis；本地load核验31/5、seed613721、
protocol/native身份、25格完整，无PMU因而不作请求预算变化结论。
response_t1_control.json保存16组合双角色共32条件记录及全部同round配对差。
沿用4T前台308..311与1T背景319、双方M48、full stripes、200ms、无gather协议。

前台16组合service delta中位范围-.83..+1.25us，相对solo约-.136%..+.101%。
双方W13且32copies时，前台solo/joint中位1242.22/1243.49us，配对delta1.25us、
MAD.27us；同前台对背景W2/32copy delta1.03us、MAD.14us。其它组合接近零或为负。
背景16组合delta范围-2.52..+7.05us；双方W13/32copy时，1T背景solo/joint中位
4930.64/4937.83us，配对7.05us、MAD.75us。中位相减不等于配对差中位。

单1T背景在此探针中属于弱响应区，不能按任务数线性扩大为32核竞争，也不能凭本场
认定每调用请求预算稳定。正式PMU将检查预算变化，2/4T对照继续原串行任务。
当前未拟合这些小量或改变模型/default，完整目标保持。

## GEMM配对1T背景第一PMU完成

本轮先核验PID647940持续推进，经同一exec74373得到t1_session1完整结束，775正式
cell记录、maxcount.025、elapsed423.451s；rsync38534 exit0回传JSONL和analysis。
decompose.py t1_session1完成992个同round role对照，first_pmu_t1.json汇总32条件。
32/32角色条件及引用对照仪器质量合格，最大PMU/control service/cycle差.621138%。

前台弱响应重复：双方W13/32copies service delta1.53us，前台W13/32copy对背景
W2/32copy为1.25us，前台W2/32copy对背景W13/W2的32copy为.62/.54us。
前台W13/32copy solo L2 refill约498836events/call，联合约498737..501654；
W2/32copy solo249603，联合249554..250111，均无显著大幅请求增加的单场证据。
前台W2/1copy solo48465，联合46373..47288，配对请求变化为-1208..-1948events/call。
该小幅下降仍受测量窗口/复用状态和重复性限制，不解释为确定缓存机制。

所有有符号预算、原始事件率及固定预算下的预期联合事件率保留；没有把小变化
拟合为通用响应，也未将L2事件转换成LLC字节。单1T背景结果不能直接推广32背景核。
原串行任务继续第二PMU及后续2/4T，完整目标和默认模型不变。

## 1T第一PMU的总DRAM固定预算检查

新增gemm_pair_response/dram_budget.py及3项测试，复用既有counter_metrics的64DDRC
事件单位和窗口归一化。每round分别扣除idle，从solo FG/BG总流量乘以各自
lambda_joint/lambda_solo并相加，预测扣idle后的总joint流量。没有将全局DDRC按角色
拆分，也没有把双方吞吐下降误算成请求量下降。原始负信号与残差保留。
3项测试覆盖共同idle扣除、双角色比例与负小信号，pytest3passed、Ruff通过。

运行dram_budget.py t1_session1，完整16组合结果和逐round原值保存在
 dram_budget_t1_session1.json。读流量残差条件绝对均值.234627GB/s，最大为前台W13/
32copy加后台W13/1copy：预测中位16.731GB/s、实际中位15.758GB/s，paired残差中位
-.973943GB/s。写残差条件绝对均值.001792GB/s，最大绝对paired残差.006724GB/s。
各分量median不保证相加，不能把这些描述性量当物理请求归因或置信界。

当前不是普遍增加DRAM预算的证据，也不能称固定预算逐点准确；需第二PMU重复。
与前台弱服务变化一起保留，不拟合统一竞争增量。exec74373继续，PID653788确认
1T第二场到round5/6，未重启或增加远程负载。

## GEMM配对1T背景两PMU完成

exec74373输出t1_session2完整结束（775cell、maxcount.025、elapsed423.603s）及
width1 complete。rsync96417 exit0后summarize.py 1生成summary_t1.json，64个role/
condition/session全部instrument_eligible。dram_budget.py t1_session2完成，不重拟合。

两场前台16组合service delta范围分别-.89..1.53us、-.59..1.78us。
双方W13/32copy的前台delta1.53/1.78us，refill配对增量1273/1317events/call，
相应solo498836/495464，弱响应可重复。前台W2/32copy对背景W13/32copy delta
.62/.48us，refill配对变化-37/+316，不能称请求增加已稳定识别。
前台W2/1copy对背景W13/1copy delta.15/.02us；solo refill48465→54750有跨场漂移，
但paired变化-1597/-1580相近，需保留同场对照，不能合并绝对预算消除状态差。

固定solo DRAM预算加双方实测调用率的读残差MAE为.234627/.280641GB/s，写为
.001792/.002669GB/s。前台W13/32copy对背景W13/1copy的读残差-.973943/-.9470GB/s
重复为负，未支持统一的竞争新增DRAM量。原始控制、两PMU、两角色事件和所有负值保留。
范围仍仅M48/单1T背景，不能替代真实32核竞争或全计划验证；未采用任何新默认。

## GEMM配对2T背景control及第一PMU完成

权威exec74373已输出t2_control及t2_session1完整结果，各775formal cell、maxcount
.0125，耗时337.549/339.237s。rsync46504 exit0回传四原始/分析文件；本地decompose
和dram_budget均成功，first_pmu_t2.json保存32role条件及其全部引用质量，32/32合格。
最大PMU/control service/cycle差1.234913%，保留小delta的测量敏感性，不据此称误差为零。
协议延续前台4T/M48和后台2T/M48、同LLC、W13/W2、1/32copy、200ms，单后台team。

双方32copies时前台service delta，control/PMU：
W13←W13 3.40/3.29us，W13←W2 2.84/2.65us，W2←W13 1.55/1.35us，W2←W2 1.40/1.22us。
与1T背景第一PMU对应1.53/1.25/.62/.54us相比增大，但整体仍弱响应，未拟合线性扩展。
前台32copy每调用refill配对变化范围约-.22%..+.75%，没有大幅预算增加的单场证据。
前台W2/1copy有-2.34%..-5.04%的配对refill变化而service delta仅.17..37us，保留
需求与服务响应的区别。第二PMU未完成，不能称跨场稳定。

DRAM固定预算残差仍有正有负（逐条件与samples保存在dram_budget_t2_session1.json），
不能用统一追加流量项解释。当前没有新的模型系数/default；原串行任务继续2T第二场
以及4T，完整动态/多team/真实计划泛化仍未完成。

## GEMM配对2T背景两PMU完成

同一exec74373报告t2_session2完成，775formal cell、maxcount.0125、elapsed339.184s，
随后width2 complete。rsync26155 exit0后summarize.py2生成64role条件汇总，64/64
instrument_eligible；dram_budget.py t2_session2完成，保留全部原始配对值。

前台32copy、背景32copy的service delta第一/二PMU：W13←W13 3.29/3.41us，
W13←W2 2.65/2.61us，W2←W13 1.35/1.34us，W2←W2 1.22/1.25us。
对应前台L2 refill每调用变化：+.688%/+.670%、+.754%/+.707%、-.224%/-.283%、
-.096%/-.324%。弱响应与小预算变化可重复，未证明多team下仍线性或固定预算。
2T固定solo预算+联合吞吐的DRAM读残差MAE .315541/.320450GB/s；双方W13/32copy
残差+.8457/+.9211GB/s，前台W13/32copy+背景W13/1copy则-.8653/-.8275GB/s。
残差符号随状态改变，不支持统一追加流量。真实宽度迁移和动态需求仍需完整验证。
原脚本继续后续4T，不改T0、系数或默认；本轮新增证据未替代原完整目标。

## 1/2T训练、4T留出的单背景响应候选冻结

M类Lab新增gemm_pair_response/fit_response.py，复用已通过解析测试的nonnegative_fit。
只读取1/2T第一场作为训练，按前台stage/copies分四组，每组两项特征：独立前台T0
乘以独立后台DRAM读MiB/us和L2 refill events/us；不使用联合事件率或阶段窗口作输入。
第一/二场完整性和全部参考质量通过；第二场评分固定第一场solo特征，只比较新y。
32copy前台W13/W2的DRAM系数为.298214982/.258687915，L2为0；1copy前台W13
为.072032045/9.916194e-7，W2为0/1.837075e-7（DRAM/L2不同单位，不能直接比大小）。
保留负y与零边界，不将其解释为多team LLC效应不存在。

freeze命令生成response_frozen_w12.json及源身份；evaluate --width1/2分别生成结果。
1T第一/二场MAE .283305/.265138us，max.902187/.602187us；2T .230905/.270108us，
max.846210/.853621us。此为低压力单background同条件重复，非新宽度验收。
两特征归一化condition约3.185，未额外搜索参数或选择后台状态。脚本Ruff通过。
MATHEMATICAL_MODEL及manifest同步，不改生产/default。

4T所有结果尚未取回或读入，reserved width4使用自身独立solo profile作输入、固定
现有系数验证joint响应，沿用阶段delta MAE5us/max12us诊断门槛，同时报告细粒度
绝对误差与零预测对照，不能因宽松门槛掩盖小量无辨识性。原exec74373继续，
PID677707确认t4_control运行；当前未宣称宽度泛化通过。

## GEMM配对4T无PMU对照完成（候选保持冻结）

exec74373输出t4_control775formal cell、maxcount .0062893、elapsed295.318s。
rsync31733 exit0回传完整control/analysis，本地load核验协议、数值/CPU与25格完整。
response_t4_control.json保留16joint双角色配对差，未运行任何重新拟合。
同4T/M48前台，背景4T/M48；stage/复用/窗口与原协议一致，主体服务响应描述如下。
双方32copy的前台delta：W13←W13 6.38us(MAD.34)、W13←W2 4.55us(.36)、
W2←W13 3.16us(.16)、W2←W2 2.41us(.15)。单份权重组合仍有接近零/负差，均保留。
这只是完整无PMU对照，不能直接证明冻结的1/2T需求响应模型推广。
原script仍运行，remote PID685108确认t4_session1 PMU已开始；需其独立solo特征
及两场质量/repeat检查后评分4T。response_frozen_w12.json未改，生产/default保持。

## 冻结响应的4T第一场宽度留出检查

exec74373报告t4_session1完整775formal cell、maxcount .0062893、elapsed297.005s。
rsync91965 exit0回传完整JSONL/analysis；features(4,1)核验全部joint/solo参考质量，
直接应用response_frozen_w12.json，未重新拟合。response_evaluation_t4_session1.json
保存16条件评分：MAE .434134us、max .898440us；零减速对照MAE1.376875us。
4T响应系数未进入训练，输入允许4T自身独立solo profile，这是单背景宽度迁移检查，
不是完全无校准或真实多team/动态泛化。

双方32copy的预测/实测：W13←W13 5.697/6.39us，W13←W2 5.532/4.64us，
W2←W13 2.442/3.34us，W2←W2 2.372/2.81us。第一场满足既定5us/12us诊断门槛，
但第二场未完成，不能据此替代重复验收和原完整目标。

decompose.py与dram_budget.py t4_session1均完成。32copy前台每调用L2 refill配对
变化范围-.565%..+.029%，后台-.564%..+.032%；请求量并无普遍大增。
总DRAM固定预算读残差MAE.583915GB/s；双方W13/32copy为+2.6441GB/s、双方W2/
32copy为-2.3054GB/s，符号随阶段改变。L2事件近似稳定不保证下层流量预算精确，
可能涉及资源路径/状态，也不能仅凭此唯一归因。原第二PMU继续，无新任务或默认修改。

## 独立系数直接迁移多team的检查（无联合响应重拟合）

将冻结的response_frozen_w12与独立gather系数直接用于已有core_budget_joint特征，
保存gemm_pair_response/independent_to_core_budget.json。后台需求固定32copy/第一场，
前台1/32copy假设均保留；gather总系数全部分给local或shared作为结构情景，不是
置信范围。uniform/prefix时序分别检查，均不使用联合y重新拟合任何系数。
仍依赖实测joint时间线与未验证长M需求外推，不是自主计划预测。

第二旧会话前台32copy：uniform下W13 MAE全local/全shared=15.605/10.029us，
W2=9.872/9.051us。同域W2 bg1/2/4预测12.04/9.46/5.49us，实际32.02/28.83/10.54。
prefix下W13=4.804/11.591us，W2=5.528/6.349us；全local时同域W13 bg1/2/4预测
47.59/43.98/38.32us，实测42.03/44.19/35.70；跨域却预测33.24/30.99/26.82，实际
24.59/23.29/22.73。同域W2 bg1/2预测25.22/17.67仍低于32.02/28.83，跨域bg4
预测12.07高于6.38。没有一个这些结构情景同时闭合所有条件，不择优采用。

该证据说明单背景独立系数尚不能直接组装多team模型；请求时序、压力幅度和局部/
共享响应划分的交互仍待辨识。不能仅靠对旧联合y回归吸收这些缺口。无新模型默认
或采集任务；原exec74373继续4T第二场，保持本轮留出候选不变。

## 单背景整批结束与4T留出重复门槛

原exec74373终止exit0，t4_session2完整775formal cell，maxcount.0063291，elapsed
296.924s，width4 complete/COMPLETE。远程ps无gemm_pair_response measure进程；
rsync53054 exit0取回最终完整数据。summarize.py4得到64/64角色条件instrument合格，
fit_response.py evaluate --width4与dram_budget.py t4_session2均完成。

冻结1/2T响应、4T第一场solo特征不变，4T第二场MAE .377835us、max1.122620us，
第一場.434134/.898440；零减速对照MAE1.457500/1.376875us（第二/第一）。
双方32copy第二场：W13←W13预测5.697、实测6.82us；W13←W2 5.532/5.01；
W2←W13 2.442/3.38；W2←W2 2.372/2.34。参数未回调。
decision.json记录passed_declared_single_background_width_gate，保持5us/12us门槛，
另报告微秒误差与零预测对照。只保留M48单背景稳态候选，不采用到production/planner。

第二场总DRAM残差仍呈阶段相关：双方W13/32copy +1.9915GB/s（首场+2.6441），
双方W2/32copy -2.4183（首场-2.3054）；不能由稳定的小L2请求变化推导下层预算精确。
三宽度控制+两PMU的原始数据全部完成且本地保留，目前没有本任务活跃远程采集。
下一步同stage/M受控增加后台压力，分离单背景到多team的响应变化与真实trace时序，
完整cost model、多team迁移、自主模拟及等预算搜索仍未闭合。

## 多team固定阶段探针实现、构建与烟测通过

E类Lab，沿用impact-analysis/test-selector/review和C++并发/平台规则。新目录
 gemm_multiteam_response的stage_service.cpp源自小型standalone，kernel/JIT ABI不改。
后台固定4T，最多8team；A/B/output每team独立，M上限96、route104，B池32copies
约3GiB总量，缩小仅探针buffer上限以控制内存。每team独立barrier、完成计数与median，
外部GO/DONE/ACK保持，CPU端点参数仅312(local)或280(cross)，至多32后台核。
前台用原narrow_stage_demand/t4/native、316..319核，与后台无重叠。

NUMA3/BF16/SVE256/H4096/F512/Ntile16，full stripes(4,0,0,1,1)，stage8/4MiB、
owner2/1MiB。前台M48 W13/W2、32copy；后台M48 W13、1/32copy、1/2/4/8team，
local/cross，加soloFG/soloBG/idle共51格。200ms窗，64DDRC+204core事件，core域
248..311与316..319，控制核240不在测量核心事件内。后台实际CPU随team数/placement
显式记录。pooled服务与per-team服务分开保存，不把aggregate计数当单team计数。

rsync83756 exit0，exec34288独立build+两smoke终止exit0；rsync79924 exit0回传原始
JSONL与build_identity。analyze.py smoke_control/smoke_pmu各51cell通过完整性、
所有team数值/CPU/输出、事件coverage、running>=99%、onecount最大.00636943。
耗时17.783/18.074秒，smoke_review线性估计正式control+2PMU、5warmup31formal共
32.359分钟（包含重复启动分配开销）。measure/events/analyze Ruff通过；不以单轮作
正式性能/PMU扰动结论。native身份在build_identity，foreground身份在protocol固定。

split.json预先规定先用冻结单背景响应评分；若增加响应项，仅count1/2/8拟合，
count4联合响应留出，可使用其独立solo输入。保留同/跨域与复用状态，paired stage
门槛5us MAE/12us max不变。独立group-solo压力诊断与单team需求聚合预测分开，
不能把组级实测压力当作任意plan输入。准备正式run.sh，模型/default未修改。
rsync53483 exit0后exec57396启动正式run.sh，control及两PMU串行，预计约32分钟。

## 多team分层诊断分析器准备

新增gemm_multiteam_response/diagnose.py，保留冻结单背景响应，不新增拟合参数。
三臂为n*单team独立rate、单team预算乘后台组独立实测aggregate调用率、后台组独立
实测rate；T0固定，DRAM共享/L2同域，逐round与前台joint-solo delta对照。
前两臂区分吞吐反馈，后两臂区分预算变化；后两者需group-solo测量，不是任意plan
可直接获取的预测输入。均保留有符号idle扣除与负delta，不作物理LLC字节归因。
质量覆盖joint、FGsolo、BGgroup-solo与BGsingle-solo及无PMU控制，两侧count和
service/cycle扰动单列；count4响应留出标记保留。

2项pytest通过，验证aggregate throughput只计一次team数、无反馈时等于独立求和；
新源Ruff通过。正式文件未完成，因此尚未执行diagnostic_session*.json，不声称
积分/响应实测验证已通过。MATHEMATICAL_MODEL及manifest同步，采集代码不变。
原exec57396仍运行；PID705478/705480核验control预热中，无新远程任务。

## 多team无PMU对照完成：前台减速未随数量线性增长

exec57396报告control完整1581formal cell、maxcount.0064103、elapsed635.226s。
rsync83149 exit0取回完整control/analysis；本地load复核31/5、seed614001、51格、
per-team输出与CPU身份。仅总结预定开发count1/2/8，count4的联合响应未查看/总结。
control_development_response.json保存24开发条件和全部配对值。

后台32copies、同域时，前台W13对1/2/8team配对delta6.27/12.13/12.48us，W2为
3.67/7.93/9.41us；跨域W13为4.04/8.33/5.94us，W2为1.70/4.04/4.00us。
后台1copy多数接近零，8team同域W13/W2为4.12/2.25us；负差全部保留。
目前仅一场no-PMU，不拟合饱和曲线，也不能以task数直接解释压力。

独立BG-only吞吐对照见control_development_background.json。32copy背景同域n1/2/8
service中位1243.53/1248.99/1253.66us，aggregate calls/s794.8/1584.5/6313.1；
组吞吐/(n*单team吞吐)为1/.997/.992。跨域对应比例1/.997/.993。
因此简单的后台调用率大幅下降不能解释前台减速趋平；仍需PMU核验每调用下层请求
与实际组压力，不能将未测流量当作已知。single_sum与group-throughput反馈应仍分开。
原control结束后脚本继续PMU，未新启实验、未改变响应系数或count4留出规则。

## 多team第一PMU完成：压力增加但冻结线性响应失效

原exec57396输出session1完整1581formal cell，maxcount.0063694、elapsed646.265s。
rsync27621 exit0回传完整JSONL/analysis；diagnose.py session1生成32joint条件诊断，
32/32仪器质量合格，所有参考PMU/control指标最大差.943501%。只查看开发1/2/8team，
count4响应未用于当前分析或改模。

后台32copy同域group-solo DRAM读率约16.00/34.28/148.07GB/s，跨域16.35/35.19/
149.72GB/s。前台W13同域delta6.51/12.39/12.10us，跨域3.86/7.75/5.11us；
W2同域3.26/7.76/10.07us，跨域1.36/3.71/4.01us。后台压力并未趋平。
冻结单背景响应在8team的W13同域误差：单team求和+33.06us、吞吐反馈+32.53us、
group-solo实测压力+39.94us；跨域分别+40.89/+40.35/+47.61us。
8team W2同域三臂+9.14/+8.91/+12.05us，跨域+15.76/+15.61/+18.61us。
因此单纯修正后台吞吐或用实测组压力替换聚合值不能修复响应，不能以task数解释。

foreground_budget_development_session1.json保存逐round前台refill预算，仅开发count。
后台32copy时，W13前台solo约498516events/call，joint各条件496881..498055，
同域8team495450，paired变化约-.64%；W2 solo246058，同域8team241119，约-1.98%。
请求量小幅下降不自动解释大幅响应差异；这也不是前台DRAM或实际exposed memory计时。
background_occupancy_development_session1.json记录后台独立DDRC occupancy/command：
同域1/2/8team约27.91/32.58/56.16，跨域27.93/32.57/55.30，代理值同样增加；
不得当作前台延迟或唯一物理归因。第二场仍待完整验证，不先拟合饱和或非单调公式。
模型系数/default未变，原脚本继续第二PMU，完整目标仍未完成。

## 多team服务中位数与调用周期检查

service_cycle_development.json对完整control/session1仅开发count1/2/8作同round比较。
32copy后台、同域W13的service delta1/2/8为6.51/12.39/12.10us，cycle delta
7.87/15.84/15.85us；W2 service3.26/7.76/10.07，cycle5.81/11.73/15.75us。
cycle含同步和整数计数分辨率，median service不是均值，两者差不能唯一归因为长尾。
W13的cycle仍趋平，约4us的口径差不足以解释约40us线性响应高估。

E类独立统计探针gemm_service_statistics/stage_service.cpp及build.sh已本地准备，
来源narrow_stage_demand原前台源码。新增mean/p90/p99/max仅在cleanup之后由既有
已完成调用duration向量计算，计时区、分配、kernel调用和数值验证源文本逐段一致检查
通过；这不是二进制性能等价证明，目标build/smoke尚未执行。分位取排序样本的
floor((N-1)*p)项，原median输出不变，空向量统计返回0。
不改正在运行的native/measure，无新增远程负载。原exec57396仍运行，PID773858
核验第二场PMU在round14/15。新增统计用于后续核验均值与累计时间，未据此改T0或公式。

## 多team全批结束，饱和候选冻结与留出结果

原exec57396终止exit0，第二PMU完整1581cell、maxcount.0063694、elapsed645.578s，
随后COMPLETE；rsync47149 exit0回传最终文件。diagnose.py session2的32条件全合格。
计数4的响应在候选冻结之前未查看；第二场结果亦未用于拟合。

M类Lab新增saturating_response.py和2项零压力/位置/有界性测试，2passed、Ruff通过。
模型delta=T0*[a_s*R/(1+R/Rc)+b_s*L/(1+L/Lc)*local]，两个共享尺度和每stage两个
非负敏感度；R/L分别为GB/s和events/us。31x31对数尺度网格[1,512]/[10,10000]，
训练loss只用第一场count1/2/8，保留负y。saturating_frozen.json冻结Rc34.296751、
Lc2511.886432，均为经验参数，不是硬件带宽或队列上限。

组级独立实测压力输入，开发两场MAE1.775/1.689us；count4留出MAE2.407/2.722us，
max5.297/6.037。进一步固定同一系数，只用第一场single-team独立profile乘team数，
不用group-solo压力/吞吐或联合时间作输入，count4 MAE3.793/4.093us、max5.935/6.275。
但是零减速对照MAE仅2.151/1.846us；同输入旧线性响应MAE9.641/9.956us。
所以改善旧模型并满足宽松绝对门槛并不足够，saturating_decision.json明确not_adopted。
count4第二场W13/32copy跨域实际.36us、预测6.397，同域实际4.54、预测9.857；
不能用单调饱和概括所有计数点，未按此留出进一步调参。

另做real_trace_worker_envelope_audit.json，检查真实core_budget_joint的阶段包络与
各worker区间。第二旧会话32x1T同域W13 envelope delta42.03us，mean-worker40.753us；
W2 32.02/30.415us。16x2T W13 44.19/40.715，W2 28.83/29.015。arrival_spread
变化约0..0.02us，故入口时差不是主要差额；worker区间仍可能包含内部同步，不能叫
纯计算。该检查未替代稳态probe均值与长尾测量。

前台统计探针已同步（rsync25922 exit0），exec92503开始独立build；尚未运行新性能
采集。原多team正式任务已经结束，下一步核验均值/分布再处理计数和压力的非单调现象。
MATHEMATICAL_MODEL与manifest同步，生产/default未改变，完整目标仍未完成。
exec92503独立统计探针build已exit0；数值烟测与均值/尾部分布采集尚未执行。

## 服务统计探针烟测通过与正式协议

取得已build的前台统计binary身份（rsync4221 exit0），在gemm_service_statistics
配置相同多team后台binary与51格，foreground仍316..319、4T/M48/32copy/W13/W2；
BG固定W13/M48、1/2/4/8team、1/32copy、local/cross，200ms、NUMA3，full stripes。
protocol固定双方binary身份。后台不重编译，前台仅测后由既有duration计算mean/p90/
p99/max，原median字段及计时区源不变；不声称编译后性能位级等价。

rsync27802 exit0，exec21898烟测终止exit0。rsync30449 exit0回传smoke/analysis，
本地load与summarize再次通过51cell/32joint、统计排序、mean有效、CPU/count/numerical
和身份检查。烟测elapsed17.783s、maxcount.0063694；两正式场投影21.339min。
measure/analyze/summarize Ruff通过。单轮不作均值或长尾性能结论。

run.sh准备两场no-PMU、5warmup31formal、seed614021/614022，保留每window的service
median/mean/p90/p99/max与cycle及同round差，分别输出差的mean和median。分位之差
不是差的分位数，不能用各分量中位数相加替代累计时间。目的为诊断此前非单调响应与
计时口径，不在本批重拟合或更换默认模型。原多teamPMU任务已结束，无并行基准负载。
rsync71192 exit0后exec1633启动两场正式统计采集，当前运行中，预计约21分钟。

## 服务分布第一场完成：均值仍有非单调响应

原exec1633报告session1完整1581formal cell、maxcount.0064103、elapsed635.383s、
32joint contrasts，继续第二场。rsync99991 exit0回传完整JSONL/analysis/summary；
本地load/summarize再次通过统计字段、协议、CPU/numerical完整性检查。

后台32copy同域，1/2/4/8team的W13平均服务配对增量（31window差的mean）为
5.864/11.732/4.332/11.648us；对应服务中位数差的median6.04/12.32/4.69/11.59us。
W2平均增量3.840/8.735/8.290/11.001us，中位数3.79/8.43/8.06/10.45us。
跨域W13平均3.787/7.958/2.911/5.981us，4team低点仍存在。因此不能把此前非单调
响应仅归于中位数隐藏长尾，也不据此重新启用被否决的饱和候选。

同域W13各档p90差中位6.16/13.47/7.97/17.98us，p99差8.65/15.86/10.66/20.90us。
它们是窗口分位之差，不是差的分位。cycle差受计数窗口影响，甚至可小于mean服务差，
不能将差额直接解释为真实同步加速。原始均值/中位/分位/最大值和所有配对记录保留。

当前只完成第一场，第二场继续；未新增拟合或改默认。后续从显式阶段重叠与实际计划
验证推进，不再仅为微秒残差添加经验曲线。该诊断未证明具体硬件/相位机制。

## 动态事件引擎增加独立refill通道

M类Lab，CodeGraph未定位ignored模拟器，按已有路径阅读capacity_simulate.py后在
独立dynamic_resource_candidate/simulate.py复用状态机。增加L2 refill预算、敏感度和
同域压力，率随实际进度变化；字节/event各自守恒，保留DAG、worker gather、阶段
切换和发布语义。事件仍是代理量，不称LLC bytes，旧logical字节也不自动变成DRAM。
可选域级refill容量为诊断性比例分配，不声称硬件公平策略；小于独立请求率时拒绝。

test_simulate.py最终9passed，覆盖解析反馈、跨域隔离、相位顺序、错峰入口、容量、
同expert自身排除及非法值。相同总量的合成相位对照完成时间17/12us只证明结构能
表达时序效应，不是硬件结果。Ruff格式/检查通过；旧通道33job结构事件/segments/
byte交付完全一致（legacy_replay_check.json），仅兼容性检查，非新模型准确率证明。
新的L2通道尚未与完整profile/计划adapter连接，不能声称planner采用或整体泛化完成。
MATHEMATICAL_MODEL和manifest同步，旧引擎、生产/default保持不变。

同一exec1633统计采集已正常结束exit0，第二场1581formal cell、maxcount.0063694、
elapsed635.168s，32joint对照完成。rsync29467 exit0回传完整数据；本地load/summarize
复核两场31/5、seed614021/22及统计/CPU/numerical身份通过。当前无活跃远程基准。
统计第二场确认同域32copy W13的mean增量1/2/4/8team为5.792/11.444/4.079/11.971us，
与首场5.864/11.732/4.332/11.648同样非单调；W2第二场3.618/8.501/6.253/10.018us，
相对首场3.840/8.735/8.290/11.001仍有场次差，保留而不抹平。均值口径不能单独
修复压力响应，下一步以动态引擎连接独立profile并做实际计划层检查。

## 独立请求接入动态引擎与窄team实际结构回放

M类Lab新增dynamic_resource_candidate/adapter.py和replay_narrow.py。适配T1/2/4、
M12..1718，使用独立请求bank的read/refill、明确copies32/session1；写请求原始值留在
raw_request_nodes，本轮read-only消融不供给write，不把负量裁剪成物理零。
时间节点12/48/96/M采用冻结isolated成本差，保留原first/later response敏感度、
gather逻辑工作和T0。首段边界/总成本不匹配、负需求差或缺失宽度均拒绝，防止
悄然借用模型。uniform/prefix两个时序是假设，不是instrumented panel数据。

最终adapter6测试+engine9测试共15passed，Ruff通过。补首段边界检查后，在所有
回放输入上做240次适配检查通过（adapter_validation.json），不改变之前数值结果。
replay_narrow.py完整运行约85.27秒，仅本地分析，无远程基准。8个窄队列条件各31
独立arrival场景，legacy/uniform/prefix三臂，自主推进gather/W13/W2/分段；原预测
逐样本精确复现。联合trace只在所有预测完成后评分，不用其时间线作输入。

第二旧会话stage delta MAE W13/W2：legacy16.089/9.663us，uniform17.063/14.116us，
prefix16.128/8.764us。任务组完成时间MAE legacy722.624us、uniform710.320us、
prefix688.947us，MAPE .585400%/.576800%/.569314%，改善很小；原首会话数据亦保留。
这不是完整MoE完成时间或搜索排序结果，因为8/16T背景及后继剩余任务尚未适配。

新refill事件已经在动态引擎积分守恒，但本轮保留原coefficients，refill响应仍为零；
只量化供给改变，不能称新LLC竞争公式已验证。legacy effective capacity与logical
gather语义明确保留，不把混合指标当物理DRAM上限。MATHEMATICAL_MODEL和manifest
同步，生产/default不变。下一步补齐完整计划宽度/小M输入并检查完成时间和排序，
不依据这次微小变化宣布模型改善或缩小原目标。

## 完整宽度输入、历史全计划回放与响应覆盖审查

补记dynamic_resource_candidate/wide_requests.py及adapter现有扩展：T8节点12/48/1205，
T16节点12/48/1718，独立bank观测不平均，插值/外推显式记录。小M默认拒绝，只有
small_m=m12_anchor才借用M12请求且保留各M自身成本；这是假设而非小M需求验证。
full_input_validation.json核对48个宽team原始观测、四臂共896次job适配，每计划224
expert、95个M<12，成本/核分配/依赖保持。当前重跑test_adapter.py、test_wide_requests.py、
test_simulate.py共18passed（.venv/bin/pytest -q，0.13s）。无新原生构建或远程采集。

已有命令`.venv/bin/python tmp/joint_cost_model_20260911/dynamic_resource_candidate/replay_full.py`
完整输出已存在，未重新启动。full_replay/summary.json保存两计划×uniform/prefix，每臂
31独立arrival预测，四臂运行40.34/40.12/35.61/36.23s。旧引擎第一场事件/完成值校验后
才预测；所有预测写出后读取plan_completion_audit/session1/2.json的每计划31个formal
样本，以compute_end_ms中位数评分。成本源expanded_joint_replay/*_expanded.json，
入口family_small_m/profile.json，开销full_baseline_candidate/results.json；均固定。

数据沿用Arm-codex-internal、NUMA3 CPU240..319、BF16/SVE256、H4096/F512，median
request016_case017 layer4、224expert，early merge off、5warmup31formal。8/16T full
stripes=(T,0,0,1,1)、Ntile16、完整W13/W2为8/4MiB，owner分别8/T与4/T MiB。

| 计划/口径 | 原需求 | uniform | prefix | 实测第一/第二场 |
|---|---:|---:|---:|---:|
| anchor完成us | 29068.013 | 28850.326 | 29201.736 | 28697.710 / 28642.220 |
| control_forward完成us | 29822.080 | 29607.552 | 29976.031 | 29650.790 / 29776.000 |
| 四项绝对误差MAE us | 253.366 | 143.102 | 397.204 | — |
| 四项MAPE % | .877 | .492 | 1.370 | — |

uniform汇总改善但control第二场绝对误差46.080→168.448us回退；prefix四项均回退。
三版均选anchor，预测control-anchor差754.066/757.225/774.294us，实测953.080/
1133.780us；未改善选择结果，且均低估两计划差距。仅两个已见历史计划，不宣称泛化。

新增full_replay/structural_audit.json保存从预测job和summary直接汇总的审查：两个计划
W13按T0累加的零shared sensitivity服务占80.077/80.014%，W2占79.598/79.527%，
新增refill sensitivity全部零。旧局部响应仍可能存在；这不是关键路径份额或残差因果
证明，但说明换需求不等于完成竞争响应。独立32copy需求到真实4copy路径迁移、小M
M12锚点、写流量省略和logical gather均未闭合。决策：保留三臂诊断，不采用或调参，
下一步先验证后续阶段响应覆盖和请求时序，再扩完整计划留出及等预算搜索。

本次为D类结果与元数据同步，L0静态检查；未修改运行时代码、生产/default或已有原始
结果。完整目标仍进行中，当前结果不满足新竞争模型或planner采用声明。

## 容量通道消融：零显式敏感度并非零竞争

进一步阅读dynamic_resource_candidate/simulate.py的fixed-point与capacity分配：
shared_capacity_kib_us=202.01615989208221会对所有正请求率的活动片段按比例降速，
不受segment.sensitivity是否为零控制。16T/W2另有local_sensitivity。因此上一节约80%
只描述显式共享响应覆盖，不能据此认定后段未建模竞争，或直接新增同量级响应系数。

本地使用相同predict入口，对full_replay四臂各固定第一个独立arrival场景，固定jobs、
T0、setup/gap/publication和其余coefficients，仅移除shared_capacity_kib_us后再预测。
脚本正常结束exit0，结果capacity_channel_audit.json；无参数拟合、原文件覆盖或新实测。

| 计划/时序 | 原容量完成us | 移除容量完成us | 变化us |
|---|---:|---:|---:|
| anchor/uniform | 28859.982 | 27533.138 | -1326.844 |
| anchor/prefix | 29208.045 | 27529.055 | -1678.991 |
| control_forward/uniform | 29612.264 | 28419.265 | -1192.999 |
| control_forward/prefix | 29982.625 | 28415.371 | -1567.254 |

prefix-minus-uniform由348.063/370.361us变为-4.084/-3.894us，说明当前两种时序假设的
完整预测差异主要经旧容量通道放大；这是模型干预，不能解释成硬件容量的实测贡献。
同时统计shared/local sensitivity都零的片段：anchor/uniform W13/W2累计高于T0
1570.504/2498.580us，移除容量后均约零（数值残差<1e-9us）。其余三臂同样成立。
累计值是各任务服务之和，不能与makespan变化相加或当关键路径分解。

现有宽team真实路径记录也限制盲目加响应：width16_response的长背景local3下，
M1718 W13增量两场12.23/17.78us，W2 8.84/8.51us且MAD较大；小/长/小队列却表现出
不同局部响应。旧整段统一敏感度及receiver16_candidate拟合均已有失败证据，不重试
同一假设宣称新验证。

下一优先级调整为需求/容量口径一致性：独立DRAM读预算与gather logical预算目前混合
求和，202.016KiB/us是旧effective参数，不能直接当物理DRAM上限。先在统一资源口径下
验证容量与后段剩余响应的分工，避免重复计费，再做完整计划留出与搜索。此轮只产出
诊断证据与D类文档更新，生产/default、T0和竞争系数不变，完整目标继续。

## DRAM容量独立记账实现与gather独立观测整理

M类Lab仅修改dynamic_resource_candidate/simulate.py并新增test_dram_capacity.py。
容量输入dram_capacity_kib_us独立于旧响应特征；GEMM dram_read_kib/dram_write_kib与
gather_dram_read_kib/gather_dram_write_kib逐worker数组显式提供。缺一任务/阶段预算
即拒绝，不将缺测当零；负/非有限预算拒绝，不裁剪。旧shared容量与新DRAM容量不能
同时开启。新预算可仅记账，未启容量时不影响原响应。旧默认接口语义保持。

新容量按显式DRAM读写率之和限制服务，只缩速正DRAM请求片段；逻辑请求再大也不
自动占用此容量。随真实预测进度积分读写并校验守恒。孤立stage或team全部gather
worker同时发出率高于容量时拒绝，避免偷偷修改T0。读写共同容量与比例策略仍未
物理标定，旧响应和新容量的联合校准也未完成，不能称真实DRAM模型已通过。

命令`.venv/bin/pytest -q tmp/joint_cost_model_20260911/dynamic_resource_candidate`
30passed（0.13s）；12项新测试覆盖跨LLC共用容量、读写总量、logical不别名为DRAM、
gather与GEMM阶段交叠、只记账不影响响应、缺失预算、非法流量、双容量拒绝、同team
孤立成本保护。Ruff格式/检查通过。无需原生构建，数值kernel和public ABI未改。
按现有完整回放源输入重新调用predict两次，anchor/control_forward首场景完成分别
29208.045373/29982.624574us，events/segment_events/delivered_kib/refill与原prefix
输出逐字段精确相等。dram_legacy_parity.json留存，仅兼容性而非新精度证明。

整理gather_demand_states/session1/2_analysis.json中仅peers=0且M>0的10条件×2场，
验证5warmup31round、每格round0..30完整、调用率正且one_call_fraction<=.05；每round
Q_kib=(rate_GBps-idle_GBps)*1e9/calls_s/1024，最后取median，避免median率之比。
gather_dram_observations.json保留20观测和各31个有符号样本、负样本计数、源路径。
未用联合场测量推算前台流量，也未将整team预算任意分摊给worker。

第一场16T无后台的读/写KiB每调用：
| M | input/output copies | 读KiB | 写KiB |
|---|---|---:|---:|
|96|1/1|26.371|.632|
|96|32/32|305.438|119.076|
|180|1/1|35.280|.693|
|180|32/32|2028.781|1145.439|
|1718|1/1|379.744|147.519|
|1718|32/32|25846.470|13860.206|

同M逻辑工作不变而物理请求估计明显随状态变化，因此不能建立单一logical→DRAM常数。
该数据为既有Arm-codex-internal、NUMA3 CPU304..319、BF16/SVE256/H4096 gather稳态
探针，100ms窗口、两场PMU、独立输入/输出轮换；实际全计划的4copy/scrub/worker到达
不同。范围仅16T/M96/180/1718且IO中间两态未覆盖1718，不直接外推所有M/T。
新通道已具备接入与校验能力；下一步将真实路径gather状态及worker预算分配闭合，再
联合标定容量和剩余响应，完整计划留出和等预算搜索仍为最终要求。生产/default不变。

## 平均容量约束审查：单阈值全成本缩放无共同可行区间

M类有界Lab诊断新增dynamic_resource_candidate/audit_average_capacity.py，不修改模拟器。
仅检验平均率模型delta=T0*max(0,Roffered/C-1)：Roffered由各状态独立FGsolo与BGsolo
读写率同round扣idle后求和，取31round中位数；T0为平均worker服务的跨round中位数，
delta为联合与solo同round服务差的中位数。没有实测联合流量进入预测压力。
分析使用已见gather_demand_states数据，不称冻结预测/新鲜验证或硬件机理证明。

在描述性epsilon=2us下，delta>epsilon的条件同时给出容量上界和下界；接近零减速
只有下界，超出容差的加速不能用正减速解释。该容差不是新采用门槛。命令：
`.venv/bin/python tmp/joint_cost_model_20260911/dynamic_resource_candidate/audit_average_capacity.py --source tmp/joint_cost_model_20260911/gather_demand_states --output tmp/joint_cost_model_20260911/dynamic_resource_candidate/average_capacity_audit.json`
完整36条件、两场5warmup31formal/round唯一性检查通过；3项解析区间测试通过，Ruff通过。

两场独立solo总读写率最高中位数134.814/134.725GB/s，来自3x8T M12/W13 BG-only；
这是已达到吞吐的描述性下界，不是统计置信界或硬件标称峰值。36条件中7个在2us容差
内需要低于该下界的容量，其中第二场M96/W13的一项仅贴边，不以计数掩盖测量噪声。
更明确的矛盾在不同形状的可行区间：

| 同一模型需同时满足 | 第一场C约束GB/s | 第二场C约束GB/s |
|---|---:|---:|
| M1718、IO32/32、W13三背景 | >=168.379 | >=168.471 |
| M96、IO32/32、W2三背景 | <=118.869 | <=119.426 |

因此该平均率/全成本缩放模型在两场均没有满足所有条件的单一C，即使暂不施加solo
下界也无解。M96/W2平均offered仅123.03/122.71GB/s，却观测2.529/2.415us减速；
用低容量放大它会严重高估大M冷状态。不能通过调一个全局容量解决这个结构冲突。

范围限制：该公式没有模拟周期内空隙、阶段突发、读写不同服务及计算访存重叠，故不能
拿此反例否定独立DRAM动态事件通道。现有gather_overlap曾在相同数据上拟合重叠参数，
其后续留出必须继续保留，不能重用旧开发点称新成功。下一步保留显式物理预算，结合
独立服务/时序证据辨识受影响部分；不继续用完整计划总时间反拟合全局容量。生产、
T0及响应参数均不变，本轮无新远端采集。数学记录与manifest同步，完整目标继续。

## 双进度动态结构实现：请求延迟可被基线工作覆盖

先核验gather_overlap_holdout的原独立留出：M180中间状态两场MAE2.2126/2.2173us、
max6.0435/6.1416us，未通过既定门槛。profile_diagnostic显示同场n0需求替换能显著
改善16/1，而成本替换几乎无效；gather_profile_history亦显示需求随conditioning变化，
且幅度跨场不稳定。因此不移植旧kr/kw、不把该候选改判通过，不在旧留出重拟合。

本轮M类Lab扩展dynamic_resource_candidate/simulate.py的显式DRAM模式：
segment.dram_service_us与gather_dram_service_us给出有效请求窗口L，基线工作T0与
DRAM服务分别推进。基线工作不因容量缩速，DRAM按Q/L和分配速度发出；两者都完成
才结束stage/worker。请求已完成则停止占用容量；计算先完成则等待请求。旧DAG、
通知/CPU释放与gather汇合状态机复用。T0不是声称独立测量的纯计算，L也未被标定。

全任务的窗口必须显式齐全、有限；有请求时0<L<=T0、无请求时L=0。孤立容量保护
使用Q/L及同team gather总率。为避免不明确的叠加，本模式拒绝任意非零旧响应系数、
旧shared容量和refill容量；旧logical/refill仅记账，不在此模式产生资源响应。
默认不提供窗口时保持原单进度路径。新功能仍是DRAM-only候选，不能称完整资源模型。

新增test_dram_clocks.py的12项测试，连同既有33项共45passed（0.14s）；Ruff通过。
解析例两任务各100KiB、L=5us、容量20KiB/us，基线分别20/5us：请求因竞争需10us，
长任务W13仍20us，短任务W13变10us；后续新到任务不会继续被已经发完请求的长任务
占用容量。另验证计算先完时后继不能提前启动、gather双进度、缺失/非法窗口和混用拒绝。
这些是合成解析结果，不是硬件微秒测量或性能提升。

对两个224-expert旧prefix计划首独立arrival重新调用predict，检查events、segments、
logical字节、refill及完成时间的逐字段一致性，结果另存clock_legacy_parity.json。
当前仅结构接入，尚无真实请求窗口/容量标定或新完整计划精度结论。下一步需要用匹配
状态的独立需求与服务观测约束L及其不确定性，再测试未见动态重叠和完整计划；不能
用最终makespan任意拟合L。生产/default、原T0与拟合参数不变，完整目标仍进行中。

## 请求窗口辨识：用独立profile选择非对称偏移探针

M类Lab新增dynamic_resource_candidate/design_window_probe.py，仅读取已有独立solo
成本与请求预算，不读取联合时长作拟合目标。gather_demand_states/session1.jsonl的
IO32/32、M96/180/1718每形状31个n0 worker向量与gather_dram_observations对应预算
检查完整；读写按joint_gather_work的逻辑份额分配是显式假设，不是逐worker物理计数。

暂取已达到的solo134.814005GB/s作诊断参考容量，换算C=GB/s*1000/1024 KiB/us；
不是硬件峰值或正式标定。每worker L_i=lambda*T0_i，则孤立总率约束给lambda下界
sum_i(Q_i/T0_i)/C。三形状下界.212983/.625719/.737969，分别与lambda=1组成两情景。
不唯一确定窗口，也不把该条件范围叫置信区间。请求状态及物理分摊的不确定性另保留。

命令`.venv/bin/python tmp/joint_cost_model_20260911/dynamic_resource_candidate/design_window_probe.py --output tmp/joint_cost_model_20260911/dynamic_resource_candidate/window_probe_design.json`
执行完成exit0：4形状组合×8偏移×4窗口组合=128次模拟，全部经过引擎T0/容量/预算
守恒检查；Ruff格式/检查通过。使用两个16T team CPU288..303/304..319，同NUMA3/LLC；
独立入口无偏差是假设，实际测量必须记录worker起止。只评分gather，两个末尾GEMM为
0.001us零需求终止占位，不是完整expert或MoE模型准确率实验。

M96/M96同步的四假设预测跨度仅.672us；M180前台/M1718后台同时启动跨度20.739us，
四种(short/short,short/full,full/short,full/full)减速为9.314/3.008/23.747/13.683us。
offset350us时对应约0/3.008/0/13.167us，可检查后台请求结束边界；offset500us均约零，
可作无重叠控制。0..200us的多个点在该假设下重复，不需要为同一信息全部采集。

下一实现选M180/M1718、0/350/500us并保留两侧独立控制；还需要原生单次调用、共享
时间基准或可验证偏移、逐worker起止、独立buffer/CPU以及扰动/数值检查。现有steady
200ms探针不能通过Python延后GO来声称测得此单次相位响应。此轮只完成独立输入的
辨识设计与执行模拟，尚未编译或采集新探针，不宣布任何硬件减速结果或参数采用。
数学模型与manifest同步；生产/default不变，完整计划泛化与等预算搜索目标保留。

## 原生单次gather偏移探针实现、构建及烟测

E类Lab新增gather_offset_probe/probe.cpp、build.sh、measure.py、summarize.py。
复用gather_overlap_holdout的API/routes头与已构建O2 gather_impl.o，不修改kernel或
生产ABI。两个独立input/output buffer组，各32copy；FG M180/16T CPU288..303，BG
M1718/16T CPU304..319，同LLC/NUMA3/controller240，H4096 BF16/SVE256。
每cell无论角色是否活跃，先FG64call后BG64call固定准备，两个角色准备期间逐次屏障。
全部worker ready后发布steady_clock共同epoch=Now+2ms；FG按0/350/500us偏移，BG零
偏移，记录各worker单次gather的begin/end。请求时序未知，实际stage重叠不等于DRAM重叠。

首构建native完成但未测量；源码审查发现线程提前退出会在另一个team计时期间发生
teardown，v2增加finished/release，所有计时调用完成后才允许线程退出和join。完成通知
及自旋仍属于探针环境，不宣称零扰动或与真实runtime完全等价。保留初构建，不覆盖二进制。
native_v2复用同实现对象重新构建，独立源码/native/object SHA记在每次protocol.json。

远端首次空闲检查因rg不可用未完成进程过滤，随后grep检查无冲突进程且新目录不存在，
才同步独立目录；无其它旧实验重启。构建命令`bash tmp/joint_cost_model_20260911/gather_offset_probe/build.sh`。
烟测命令`.venv/bin/python tmp/joint_cost_model_20260911/gather_offset_probe/measure.py --output tmp/joint_cost_model_20260911/gather_offset_probe/smoke --seed 614101 --rounds 1 --warmup 0`。
exec48242正常exit0，7cell、elapsed3.618s、timing_bad0。数值/padding/guard、32worker CPU、
目标时刻<=begin<end与inactive未执行计时调用检查通过。数据已取回，Ruff及shell语法通过。

正式协议7cell：三个FGonly偏移、三个joint偏移、一个BGonly；每场5warmup31round，
seed614121/614122，随机cell顺序，全部原始记录保留。时序质量预设每worker迟到<=10us、
每team start跨度<=5us；失效样本标记，不删样本后声称通过。summarize按同round相同偏移
FGonly作配对基线，同时报告平均worker服务、包络、实际start差及stage重叠。
无PMU，单次时序探针不接入counter窗口；不将旧稳态IO32/32成本自动当本批无竞争真值。
exec97310已启动两场串行正式采集，目前结果待完成；预计数分钟，未修改模型参数或默认。

## 原生偏移两场结束：350us仍有减速，500us回到近零

exec97310最终exit0，两场各252cell、elapsed109.284/109.284s，均完成数值、保护区、
CPU与原生记录完整性检查。session1 timing_bad0；session2 timing_bad2，分别为BG-only
warmup round-3跨度8.99us、formal round18跨度5.96us，超过5us门槛（迟到仍<=10us）。
全部FGonly/joint正式配对时序合格，不将BG-only异常隐藏或删除后报告全通过。
两场protocol/stdout/stderr/analysis已rsync回本地，来源及native/object身份保留。

summarize.py按同round、相同offset的FGonly与joint计算差，31pair全保留。初summary
只标记前台配对质量；补充独立background_reference_timing_valid后另生成summary_v2，
原文件不覆盖。正式BG-only参考异常在第二场各offset汇总中均显式标记1。

| 前台偏移us | 第一场配对增量us | 第二场配对增量us | FG stage与BG stage重叠比例中位数 |
|---|---:|---:|---:|
|0|7.626|7.819|约1|
|350|9.270|9.655|1|
|500|-.023|.159|0|

第一场三档FGsolo平均worker服务中位数41.587/41.313/41.214us，第二场40.985/40.801/
40.926us；BGsolo405.979/404.484us。实际起始偏移中位数第一场-.06/349.98/500.02us、
第二场0/349.99/500.03us，支持本批原生偏移确实实现。start/end全向量保存，不拿
请求的offset代替实际时序。第一场worker增量MAD .519/.376/.237us，包络增量亦同方向。

temporal_contrast.json进一步保存同round的(delta350-delta0)，中位1.823/1.656us、
MAD .870/.671us；不是两个独立中位数直接相减。晚期响应略强且两场同向，不能用
“后台仍在算，但其请求早已全部结束”解释本组350us响应。这里仍只有服务和stage时间，
未直接测DRAM/LLC或证明唯一硬件原因。旧short情景约零不符合本批晚期服务增量，
但long情景也未被数值验证：本批准备、单次与steady profile存在差异，不能直接采用参数。

命令为`measure.py --output .../session1 --seed 614121 --rounds 31 --warmup 5`与第二场
seed614122；评分为`summarize.py .../sessionN.analysis.json --output .../sessionN.summary_v2.json`。
NUMA3/CPU、H4096/SVE256、独立buffer、64call准备和2ms epoch协议均同上一节。
无PMU、无模型重拟合；完整采集已终止，不追加样本追逐质量或调整门槛。
下一步将此晚期响应约束与状态匹配/同域跨域对照结合，辨识需求时序和服务通道，再
接入完整计划；生产/default与已冻结成本不变。数学记录/manifest同步，完整目标仍进行中。

## 同域／跨LLC单次偏移对照准备与启动

E类Lab新增gather_offset_locality，保留gather_offset_probe全部源码/二进制/原始结果。
仅将后台起始CPU作为显式命令参数：FG固定288..303，BG local304..319或cross256..271，
同NUMA3、相同独立buffer/shape/32copy/64call FG后BG准备/2ms epoch；每种placement
各自FGonly三个offset及BGonly控制，因此不会用旧二进制同域时间与新跨域直接相减。
kernel实现与对象不变；新native在命令、Pin、CPU校验和JSON中显式记录bg_begin。

初次本地生成器的字符串断言错误发生在写源文件之前，仅创建空目录；修正断言后生成，
未启动错误版本或覆盖已有实验。远端空闲/新目录检查后同步、构建并烟测，exec31834
exit0，14cell数值/保护区/CPU通过但BG-only两格timing超界：local跨度12.86us、cross
6.01us（门槛5us）。未放宽门槛或直接进入正式采集。

源码审查发现inactive FG的完成计数更新与其它原子状态可能同cacheline；v2将ready、
finished、release、epoch分别alignas128，旧源码另存probe_smoke_v1.cpp、原native与
smoke不覆盖。exec90054构建native_v2并执行seed614202单轮烟测，exit0，elapsed6.671s，
14cell timing_bad0，全部数值/CPU/时序检查通过。一次烟测不证明旧异常唯一由false
sharing造成，也不声称跨版本性能等价。只在新版本内配对比较正式local/cross。

正式14cell×(5warmup+31round)每场504调用，两场seed614221/614222串行，约7分钟。
exec1398已启动，当前第一场运行中；无其他基准同时运行，不改模型系数。design.json
记录主要指标为350us同round(local paired delta-cross paired delta)，并报0/500us、
两侧solo/joint服务及实际stage重叠。所有时序异常保留，最多两场，不追加样本追逐显著性。
summarize.py继承原配对统计函数并显式按placement分组；Ruff、shell语法、14cell烟测
协议本地复核和diff检查通过。尚无正式local/cross结果或LLC/DRAM唯一归因。

## 同域／跨域正式两场完成：跨域晚期响应保留，局部差额不稳定

继续轮询同一exec1398并只读核对PID927330/927331与session1.stdout进度，没有重启。
两场最终正常exit0，各504cell、elapsed218.105/218.154s。第一场timing_bad2：warmup
round-4 local joint offset0的FG迟到18.73us/start跨度12.17us；formal round1 local
FGonly offset0迟到14.62us/span14.52us。第二场timing_bad0。数值/保护区/CPU/条件
与记录完整性均通过，失效时序记录没有删除；第一场0us local配对含1个质量异常。
两场protocol/stdout/stderr/analysis已收回，summarize.py各生成31正式配对/六组结果。

| 偏移us | 第一场local增量us | 第一场cross增量us | 第二场local增量us | 第二场cross增量us |
|---|---:|---:|---:|---:|
|0|7.634|4.131|7.071|4.146|
|350|6.359|5.396|8.539|4.752|
|500|.231|.486|-.011|.319|

0/350us阶段包络重叠中位数约.99..1，500us均0。表为各自同round、同placement、
同offset n0扣除后的平均worker服务差中位数，不是端点差或纯DRAM服务。
同round再比较local-minus-cross：0us为3.121/2.992us（MAD1.398/.932），350us为
1.047/2.963us（MAD2.142/2.432），500us为-.264/-.185us（MAD.537/.601）。
不能用表中两个独立中位数相减代替这些配对结果。

350us FGsolo第一场local/cross为42.323/38.994us，第二场41.957/38.610us；BGsolo
第一场local/cross386.568/392.077us，第二场380.698/390.708us，均保留。前台核未变，
准备/后台放置流程却影响FG无后台成本，提示状态匹配仍重要；本批各自扣n0避免直接
混为竞争，但不能因此断言两位置物理请求预算相同。

两场都支持跨域仍存在明显晚期减速，不能把350us响应只写成同LLC竞争。局部额外
响应有方向但幅度尤其350us不稳定，暂不拟合一个固定local系数。该对照只有stage时序
和服务，不能唯一识别DRAM、LLC或请求结束时间。相同准备流程内的无后台预算应在
同一单次探针中独立测量，才可把需求变化与服务变化分开；旧steady profile不自动迁移。

本轮没有改模型参数或新启其他实验，两场已结束；下一步准备匹配单次状态的PMU
独立预算测量，并检查counter覆盖与非PMU控制扰动。不能将每controller不同计数窗口
的平均GB/s直接乘一个统一窗口来推单次字节，应保留各event计数/时间并分别扣idle。
原native/object/source身份、14cell协议、CPU/NUMA/64call准备及2ms epoch见本批protocol。
Ruff与diff检查通过，manifest/数学记录同步，完整计划泛化和等预算搜索仍待完成。

## 单次gather独立DRAM预算：计数窗口实现与正式标定启动

E类Lab新增gather_offset_pmu/probe.cpp、build.sh、measure.py、analyze.py及计数测试。
复用locality_v2的独立buffer/CPU/准备流程及同一gather_impl.o，计时kernel调用不变。
原生准备完成后ARMED，控制器启完64个NUMA3 DDRC event再GO；epoch固定为原生
ready+2ms。全部worker调用结束后DONE，控制器停完计数再ACK，之后release/join与
数值/保护区检查。失效GO/ACK仍释放线程并报告错误，避免异常路径留下joinable线程。

measure.py将Pythonmonotonic的enable-done和disable-begin与各worker原生时间比较，
保证全部计数器完整覆盖单次调用；每window记64个独立count/enabled/running，RESET
仅清count、时间差扣前一次累计值。active角色限制为FGonly/BGonly/idle，不把joint
aggregate冒充某角色预算。全程无并发其它基准，原始stdout结果写JSONL且已有文件拒绝覆盖。

analyze.request_budget按event分别计算(count_active-count_idle*enabled_active/enabled_idle)
再按flux32bytes换算KiB求和，保留有符号结果；不能把各controller的平均GB/s乘同一
公共窗口或假定所有output已同步写回DRAM。2项单测验证不等计数窗口和负写残差、缺失
计数拒绝，2passed；Ruff及shell语法通过。该预算是观测窗口归属，不是逐worker物理归属。

exec51757构建native及无PMU/PMU各10cell烟测正常exit0，数值/CPU、原生时序和计数
包围检查均通过。单轮不同进程PMU/control部分服务差>5%，不能据此称扰动合格或唯一
归因PMU。随后只改Python在同进程同buffer随机交错PMU开关，native不重编译。
exec87126的16cell smoke_paired（seed614303，1round0warmup）也正常结束，计数/
时序检查通过；单轮BGcross开关差-8.44%、FGcross350差+5.44%仍保留，不靠烟测调参。

正式设计六个active条件：FG M180/16T CPU288..303、offset0/350us；BG M1718/16T
local304..319或cross256..271。每条件PMU开关各一次；四个placement/offset匹配idle
只需PMU，没有可比较的idle kernel service。共16cells×(5warmup+31round)每场576。
两场seed614321/614322，--paired-control，同进程同buffer，native/object身份各场记录。
design.json在正式结果前固定各条件abs(median paired service pct)<=5、running>=.99、
worker迟到<=10us/start span<=5us，原始时序失效与负预算均保留，失败条件不直接进模型。

exec80046已启动两场串行正式标定，预计8–9分钟，当前尚无正式结果。不补额外样本
追逐通过、不更改T0/容量/响应参数；完成后先检验扰动与跨场需求稳定性，再用于窗口
辨识。新单次数据不等同真实MoE route状态，完整计划泛化和等预算搜索目标保留。

## 单次DRAM两场完成：仪器门槛通过，匹配状态需求差异重复

继续同一exec80046，第一场完整后取回分析，第二场保持原协议；最终正常exit0。
两场各16cell×36=576记录，64event完整包围计时调用、running>=.99、数值/保护区/CPU
检查通过；全部formal PMU与control时序通过。各6个条件abs(median paired service pct)
最大.627%/.390%，均低于预设5%门槛。单轮烟测的较大差没有被删掉或当作正式通过，
正式使用同进程、同buffer随机交错控制，原始session1/2 JSONL均保留并已收回。

新增quality.py拒绝服务偏差超界、任一侧时序异常或负预算中位数直接进入匹配profile，
保留原始有符号样本；3项新资格测试加2项计数测试共5passed。contrasts.py按同round
计算差并携带两侧资格，不把两个独立中位数相减冒充配对结果。

| 单次独立状态 | 第一场读/写KiB | 第二场读/写KiB |
|---|---:|---:|
| FG180，local准备，offset0 | 2710.875 / 1118.171 | 2509.022 / 1078.896 |
| FG180，cross准备，offset0 | 1758.619 / 1169.686 | 1696.244 / 1090.985 |
| FG180，local准备，offset350 | 2718.752 / 1121.929 | 2525.054 / 1098.490 |
| FG180，cross准备，offset350 | 1794.651 / 1223.547 | 1687.401 / 1072.365 |
| BG1718，local位置 | 23402.009 / 16218.911 | 23444.311 / 16203.679 |
| BG1718，cross位置 | 25708.646 / 15177.817 | 25871.958 / 15834.093 |

同round FG local-minus-cross read差offset0为939.433/814.889KiB，offset350为
921.973/861.132KiB；对应服务差3.483/3.031us与3.483/3.184us。方向重复，但幅度
仍有跨场差异。FG write配对差offset0为-89.588/-21.715KiB，offset350为-134.641/
+30.150KiB，不能用同一个“local放大倍率”同时表述读写。各placement中FG350-0
读差仅约.4..16.4KiB，远小于本批local/cross差；这只是本组匹配状态，非所有等待时长规律。

该结果提供了“形状相同但需求不同”的直接独立计数证据。尚不能唯一分解A读取、RFO、
cache驻留或写回，也不能把counter观测窗口全部write预算强制当成stage结束前同步需求。
因此此前local/cross服务差不能全部被写成接收敏感度，应先输入状态匹配的需求和基线。

build_profiles.py核对quality/analysis/raw来源、31round与每个合格状态，将无PMU的16
worker成本与同round PMU请求预算配对，生成matched_profiles.json：12profile、372观测。
两session独立保留，物理worker分配明确unassigned，不插值未知M/T/历史，不改任何
已有T0或响应系数。文件family限定native_single_gather_fg64_bg64_ready2ms_io32。
命令为analyze.py sessionN.jsonl、quality.py --design design.json、contrasts.py及
`build_profiles.py --directory .../gather_offset_pmu --output .../matched_profiles.json`；
完整CLI和源码保留本目录，Ruff/diff检查通过。两场已结束，后续用这些显式状态预算
约束窗口，再做新的联合验证；完整计划泛化和等预算搜索仍是最终目标。

## 匹配状态预算接入双进度：起始拟合好，晚期仍低估

M类Lab新增dynamic_resource_candidate/fit_matched_windows.py。读取12个matched_profile，
各worker成本取无PMU控制的中位数，读写量取独立计数中位数，保持整个网格不变。物理
预算按joint_gather_work逻辑份额分配是显式未验证假设。每role令rho=sum(Q_i/T0_i)/C，
L_i=T0_i*(rho+theta*(1-rho))，rho>1拒绝；theta=0在孤立聚合容量边界，theta=1为整段。
容量140/170/200GB/s是有限诊断网格，不是硬件标称/实测peak；前后台theta各五档。
旧响应全部关闭，只有DRAM容量与双进度。500us FG使用350us profile作无重叠控制，
不冒充该时刻已采需求。终止GEMM占位仍不评分，不称完整expert/MoE实验。

命令`.venv/bin/python tmp/joint_cost_model_20260911/dynamic_resource_candidate/fit_matched_windows.py --output tmp/joint_cost_model_20260911/dynamic_resource_candidate/matched_window_fit`
完整75参数组×2session×2placement×3offset=900次模拟，14.314s，先保存全部predictions
再读旧联合结果，仅第一场offset0 local/cross两点选参数。准备/driver/process之间的
profile迁移限制保留；联合数据此前已看过，不声称盲测、预注册预测或模型泛化通过。

选择C170GB/s、theta_F1、theta_B.75，训练MAE .282565us：
| 条件 | 第一场预测/实测增量us | 第二场预测/实测增量us |
|---|---:|---:|
| local offset0 |7.579 / 7.634|6.883 / 7.071|
| cross offset0 |3.621 / 4.131|3.669 / 4.146|
| local offset350 |1.835 / 6.359|2.163 / 8.539|
| cross offset350 |1.712 / 5.396|1.975 / 4.752|

500us四条件预测约零，保留实际微小有符号差。晚期明显低估，不能用起始训练误差好
宣布参数可采用。identifiability_audit.json保存训练MAE距最优<=1us的四组参数，
它们的350us预测均低（多数约零），不只是某个任意tie-break造成当前失败。

同一已保存网格的事后开发审查：若第一场0/350四点均用于选参，最优C140、theta_F.5、
theta_B.75，第一场MAE1.795us、max3.962us，第二场四点MAE1.155us。它用到了已经
评分的晚期开发数据，不再是350us留出，且存在起始/晚期折中，不作为新采用模型。
这是有限网格结论，不据此否定所有连续参数或所有重叠模型；单一均匀请求窗口和纯
容量约束是否充分，仍需与时变请求密度、容量以下服务延迟及profile迁移分别验证。

test_matched_windows.py两项测试验证窗口端点满足孤立总率约束、不可行容量不裁剪，
2passed，Ruff通过。原引擎/源数据/默认模型未改，预测、评价和参数非唯一性诊断均保留。
下一步围绕上述两种机制用独立证据约束，而非按每个坏case添加窗口参数；真实完整
计划预测与等预算搜索仍未完成，目标保持。

## 联合单次计数与command占用代理：探针扩展并启动

先对已有gather_offset_pmu两场solo数据进行event-specific idle扣除，保存
gather_joint_pmu/solo_occupancy.json。12条件的corrected occupancy总和/read-command
总和均有正有效信号；FGcross约51.30..51.73、FGlocal54.24..54.61，BG60.80..61.92，
MAD约0.44–0.77。它是aggregate command加权代理，不是前台CPU延迟，不换算为us，
也不能只凭独立值认定联合排队响应。

E类Lab新增gather_joint_pmu：仅放开原生fg&&bg条件，保持独立buffer、CPU、M180/1718、
16T、64call FG后BG准备、ready+2ms epoch和ARMED/GO/DONE/ACK。kernel/object不变。
新增local/cross joint0/350us及其PMU开关对照，共24cell；独立角色和idle均保留，
所有active条件同进程同buffer随机交错PMU开关。不存在以不同二进制替代本批n0的问题。

exec39134构建并执行seed614401、1round0warmup的24cell smoke，正常exit0。
数值/保护区/CPU、时序及全部counter包围检查通过。单轮某些服务开关差仍较大，
只作为执行有效性检查，不判断扰动、流量可加性或模型采用。原始smoke已收回，不调参。

analyze_joint.py复用per-event计数校验/idle校正，保存两role的PMU及无PMU配对时差、
joint与solo开关偏差、read/write残差和command占用代理差。一次调用的联合预算残差
为Qjoint-Qfgsolo-Qbgsolo，不套steady throughput因子。独立占用对照为
(Ofg+Obg)/(Nfg+Nbg)，不是两个比值的简单平均。2项测试验证不同event窗口和不同
controller权重、非正command信号返回missing而非零延迟，2passed，Ruff通过。
任何联合残差都不唯一分配给FG/BG，负时间差/负流量残差完整保留。

design.json固定两个session、seed614421/614422、各24×(5warmup+31round)=864调用，
全部串行，预计12–13分钟。正式各条件两role的paired PMU/control偏差中位数绝对<=5%
及running>=.99、时序10/5us门槛保持；不增加样本追逐效果。exec17332已启动，目前
第一场运行中。目标是检验需求变化与服务代理变化，尚无正式结论或新cost参数。
manifest/数学记录同步，生产/default不变，完整计划与等预算搜索目标继续。

## 联合PMU两场完成：中心值掩盖大幅请求状态变化

同一exec17332两场最终exit0，各864调用、5warmup31round，全部原始记录已收回。
analyze_joint.py与新增assess_joint.py检查前后台joint/solo四类PMU偏差、时序、预算
中心和代理有效性，8组均通过既定测量门槛。5项计数/质量测试通过；负服务变化、负
流量残差和负代理差不会被当作质量失败或截零，测量合格不等于模型假设成立。

| 条件 | 第一场FG control增量us | 第二场FG control增量us | occupancy代理增量第一/第二场 |
|---|---:|---:|---:|
|local0|6.071|6.692|5.491 / 4.537|
|local350|8.056|6.458|5.130 / 5.092|
|cross0|4.393|4.617|3.979 / 3.408|
|cross350|5.211|5.241|5.030 / 4.284|

BG control增量第一场3.306/6.435/1.169/3.669us，第二场4.374/4.886/5.124/5.173us。
PMU增量另存，不能将独立比值中位数相加替代差值；也不把整体command代理指定给FG。

初看read/write残差的有符号中位数较小，但additivity_variation.json的完整逐轮审查
发现read绝对误差均值21.2..32.5%、P90 53.0..70.4%（nearest-rank）；write绝对误差
均值约3.0..5.4%、P90约5.8..10.5%。读中位数约-1.3%..+.35%只是正负抵消，不能
宣称固定独立预算在逐次调用上可加。所有31pair保留，未修改既定质量门槛。

background_state_modes.json按独立BGsolo读量排序的最大间隙作描述性分组：
| 场次/位置 | 低读组数量/读KiB/服务us | 高读组数量/读KiB/服务us |
|---|---|---|
|1/local|12 / 14009 / 378.951|19 / 23296 / 415.642|
|1/cross|19 / 14475 / 388.724|12 / 25739 / 426.443|
|2/local|12 / 14039 / 379.362|19 / 23261 / 412.279|
|2/cross|13 / 14569 / 386.616|18 / 25888 / 422.303|

组间读间隙8.1..10.2MiB，服务差33..38us。只证明此数据存在分离的请求/服务状态，
不唯一归因cache/RFO/频率或竞争。state_variation_audit.json还保留事后read残差<=5%
的11..17/31子集，occupancy增量中位数仍3.408..5.990；这是条件诊断，不是删异常后
重新宣布预算可加或通过验收，也不能据此直接拟合队列延迟。

build_profiles.py生成same_call_profiles.json：12profile/372观测，明确同调用字段为
pmu_worker_service_us、request_budget_kib、occupancy_per_command，control_worker_service_us
属于另一调用。cost_request_pairing_audit.json用读组对应服务中心的中点作描述性
阈值，14..17/31的PMU/control配对位于相反时间组。control没有读计数，时间组不是
其真实请求状态证明，但不能把round配对误当同一状态的成本/需求实现。
旧matched_profiles和旧窗口拟合不覆盖；后续需保留状态及同调用关联，避免独立取
中位数拼出不存在的状态。该新bank不自动替换冻结T0或进入planner。

## DRAM服务响应结构接入，系数保持未标定

M类Lab在dynamic_resource_candidate/simulate.py新增可选dram_queue_us_per_kib。
双进度模式按其它expert的实际DRAM请求率求有效服务速度1/(1+k*P_other)，与容量
一起固定点求解；自身整个team排除，零/已完成请求不受影响，基线工作时钟不缩速。
旧logical/refill响应仍不允许混用。该系数不是由occupancy代理换算的真实latency。

5项新测试覆盖未触发容量时的解析反馈、可隐藏的请求延迟、同team不重复计费、k0
事件等价及legacy输入拒绝；连同47项既有测试52passed（.23s），Ruff通过。仅结构
候选，k默认0、没有拟合任何值。鉴于上述请求状态变化，当前优先处理状态/协方差，
不以中心代理增量直接调k来吞掉预算错误。两场硬件采集已结束，生产/default未变。
数学记录/manifest同步，完整计划泛化与等预算搜索目标保持。

## 独立状态分布对照：量级可迁移，比例仍有差异

M类Lab新增gather_joint_pmu/state_distribution.py。仅第一场same_call_profiles的solo
FG/BG各31观测构造31x31独立组合；每role内部read/write配对保留，role间独立是假设。
所有预测先保存state_distribution_audit/predictions.json，再读取两场joint评分；不按
联合调用的实测状态匹配预测。未拟合新系数，数据此前已看过，不称新鲜独立验收。

归一化W1（除实测均值，单位%）第一场read四条件local0/local350/cross0/cross350为
11.981/7.403/2.098/5.355，第二场7.565/4.856/10.023/5.117；write为约.831..3.181。
这不是逐轮残差MAE，不能与前一轮21–32%误差直接相减宣称模型改善。
预测低读组比例local12/31、cross19/31；joint实际对应第一场20/17/18/16，第二场
17/15/13/16（每格31），说明整体分布的权重存在差异，但可能包含有限样本波动和
状态变化，未作因果认定或统计显著性声明。

conditional_levels.json按预测最大间隙固定阈值，事后分别描述实测joint低/高组。
组内read分布W1约.185..2.181%，多数低于1%；支持已测组的量级较接近，主要余差
涉及组比例。这些组使用了joint计数，不能作为planner预测输入、硬件状态标签或新的
通过率筛选。3项W1解析测试验证不等样本数、位置平移和非法输入，3passed、Ruff通过。

## 同调用状态传播到动态时间：晚期低估仍存在

state_time_replay.py保持原窗口候选C170GB/s、theta_F1/theta_B.75、queue0，第一场
每role同一次PMU的16worker成本与预算成组传播，不独立取中位数拼状态。四个条件
各31x31场景，共3844分布预测，另有4个marginal-median输入对照；49.279s完成。
使用逻辑份额分配物理字节、假设role间独立，终止GEMM占位不评分，不称完整MoE。
第二场不更新profile；预测写出后分别评价PMU及无PMU的实际joint服务分布。

| 无PMU评分项 | 第一场local/cross误差us | 第二场local/cross误差us |
|---|---:|---:|
| FG0 均值 | -1.611 / -2.810 | -1.065 / -2.158 |
| FG350 均值 | -6.260 / -4.643 | -5.218 / -3.767 |
| BG0 均值 | +2.981 / -7.934 | +6.828 / -6.289 |
| BG350 均值 | -3.291 / -9.793 | +3.031 / -3.743 |

状态传播保留了低/高服务组，但未修复FG晚期系统低估。BG中位数误差仍可达
-37.734/+30.537us，与均值表现明显不同；marginal-point也未消除此问题。各场/各role
的mean、median、P10/P90、W1及单点对照完整保存在state_time_replay/evaluation.json，
不能只选较好的均值或分布指标替代原验收。参数没有重拟合，不能称新模型已采用。

本轮是显式输入状态修正的消融，没有新硬件采集或默认修改。下一步用同调用状态
检验服务响应项对晚期误差的作用，并保留状态权重与profile迁移限制；完整计划预测
和等预算搜索仍是最终要求。Ruff/diff检查通过，数学记录与manifest同步。

## 服务响应候选校准：晚期改善，容量和后台中位数仍未闭合

M类Lab新增fit_state_service.py；state_time_replay.evaluate_pair显式传递可选queue
参数，缺省0保持原行为。新增plumbing测试证明非零参数确实到达solver，缺省与零
完全相同，1passed；Ruff通过。没有改kernel或默认planner。

训练仅第一场：用独立FG读量三等分、BG最大读间隙两组，各组选择真实调用代表与
经验权重；成本/请求仍同调用。预测组由独立输入总读量和此前独立分布阈值确定。
训练目标为第一场PMU joint的低/高读组中FG/BG平均服务，实际joint计数组只参与
训练损失，不作为前向预测特征。第二场没有用于选择参数，但数据之前已看过，所以
这是回顾校准/迁移诊断，不能称新鲜独立验收。

容量170/200/230GB/s、theta_F=.75/.875/1、theta_B=.9/.95/.975/1、k=0/.001/.002/.004
共144组，用代表性状态搜索。选出k0和k>0各自最优候选，再各跑4×31×31完整状态
组合；总耗时409.415s，其中代表性搜索146.694s。未用目标状态重新加权预测分布。
完整输出service_fit/search.json、finalist_0/1.json、evaluation.json均保留。

| 比较 | 容量-only候选 | 服务响应候选 |
|---|---:|---:|
| 参数(C,theta_F,theta_B,k) | (170,.75,1,0) | (170,1,1,.002) |
| 代表性状态训练MAE/us |4.484|1.484|
| 完整状态训练MAE/us |3.957|1.608|
| 第二场条件诊断MAE/us |3.546|2.350|
| 第二场条件最大误差/us |6.858|6.509|

conditional_evaluation.json中的第二场分组使用实测joint计数，只是分项诊断，不能
拿它代替无条件预测。第一场是训练数据。窗口与响应在两模型里各自重拟合，改善
不全归因单独k；已有代表性状态2x2固定C/BG窗口对照另存coarse_factorial.json，
不把该近似对照冒充完整分布因果分解。

无条件、无PMU的实际评分仍使用第一场独立状态权重：
| 指标（每场8个FG/BG条件） | 容量-only第一/第二场 | 服务响应第一/第二场 |
|---|---:|---:|
| 各条件均值的绝对误差MAE/us |5.623 / 5.543|2.921 / 3.566|
| 各条件中位数的绝对误差MAE/us |16.521 / 13.989|13.112 / 11.683|

服务响应版第二场FG均值误差local0/local350/cross0/cross350分别+.491/-.490/+2.077/
+1.245us，对照为-5.260/-5.141/-3.584/-4.083us。晚期系统低估改善，但BG中位数
仍受状态权重影响，不能用均值或条件内结果替代原完整计划/选择门槛。

另发现最佳代表性theta_F1/theta_B1/k.002下，C170/200/230训练误差完全相同。
当前只能支持进一步验证服务响应，不能把170解释成辨识出的硬件容量；高并发计划
仍需要独立容量约束。候选暂不采用，下一步冻结后验证未见条件，并回到完整计划与
等预算搜索；本次无硬件采集、生产/default与原始结果未变，数学记录/manifest同步。

## 独立读容量扫描闭合：不能继续把170GB/s解释为通用容量

复核dram_capacity_probe的两个正式session，seed614521/614522，各5warmup/31round、
17cell/round，终止记录完整，耗时173.260/173.279s。当前Arm NUMA3，M1/1T、
H4096/F512、Ntile16、full owner stripes(1,0,0,1,1)、四份独立权重；20/40/60/79team，
控制核240，worker从CPU320向下分配。每cell200ms，PMU/control同进程配对，
64 DDRC计数按event归一化并减matched idle。此轮复核没有重新启动硬件实验。

| 阶段 | team数 | 第一场读GB/s | 第二场读GB/s |
|---|---:|---:|---:|
|W13|20|175.723|177.043|
|W13|40|274.067|273.749|
|W13|60|289.126|290.497|
|W13|79|294.217|294.177|
|W2|20|161.261|161.111|
|W2|40|268.070|267.738|
|W2|60|286.891|286.930|
|W2|79|292.534|292.559|

两场各8组均通过预先定义的仪器扰动/计数门槛，最大one-call fraction分别.01087/.01111。
同轮60→79读率增幅W13为1.746/1.326%（MAD .069/.154个百分点），W2为
2.001/1.967%（MAD .167/.112）。四项均满足预先的绝对中位增幅<=5%描述性平台标准。
写流量中位数仅约.08–.24GB/s，因此这是读主导负载的平台，不是混合读写容量或硬件峰值。
增加team也改变cache驻留和placement，不能单独证明controller饱和机制。

plateau_decision.json保存分析文件SHA、全样本质量结论和平台判据。实测读率已超过
170/200/230GB/s，这三者不能作为该域通用物理总容量；旧拟合仍作为历史控制保留。
下一消融冻结窗口/queue参数，仅比较旧总容量与独立读约束，不借此重新拟合吞掉差异。
新鲜M/offset、完整计划预测和等预算搜索尚未完成，默认planner不变。

Lab模拟器已有独立dram_read_capacity_kib_us通道，本轮完成复核与登记：只约束
实际读请求率之和，不将write-only actor计入读容量。读写预算仍分别守恒，写可以
参与已有queue响应或另行标定的total cap。读cap要求完整物理预算与独立请求率可行；
不与旧logical cap混用。它并不表示真实读写互不干扰，混合服务曲线仍未标定。

复核命令`.venv/bin/pytest -q tmp/joint_cost_model_20260911/dynamic_resource_candidate`：
56passed，.18s，包含4项read-cap解析/拒绝输入测试。没有设置新的默认容量数值，
没有kernel/生产接口更改。本次同步仅增加证据产物与Lab文档，保留既有用户改动。

## 固定窗口与queue的容量消融（运行中）

新增Lab脚本gather_joint_pmu/capacity_ablation.py，未修改既有预测器。运行命令：
`.venv/bin/python tmp/joint_cost_model_20260911/gather_joint_pmu/capacity_ablation.py`。
测前protocol.json冻结第一场same-call profile及finalist_1文件SHA，FG/BG theta均1，
queue=.002不变；四组为旧total170、read292、read295、no-cap。292/295只包围已测读平台，
不解释为mixed容量。每组四条件各31x31独立输入组合，共15376场景；不读取目标状态。
旧total170逐样本须复现既有finalist，最大差<1e-9us，否则中止。预测保存后再读取
两场joint分别评分PMU/control均值、中位数和W1，数据已见，非新鲜验收。Ruff通过。
exec session24787已启动，首个local0条件完成：四组逐样本时间相同，尚不能外推其它条件。
无硬件重测或默认参数修改；等待同一handle完成后补充结果，不能重复启动覆盖产物。

## 完整物理模型接入审计

只读既有full_replay/anchor_prefix和control_forward_prefix，保存源SHA到
full_physical_readiness.json。两计划各224expert，均95个M<12；这些expert占累加
GEMM T0约7.17%，此比例不是关键路径贡献。每计划788个阶段片段仍使用旧read_kib字段，
没有完整physical read/write/window输入，224个gather也均缺三类physical输入。

原始请求节点保留写预算，计划合计42.933/42.928MiB，读3512.494/3523.456MiB；
逐前缀写增量均非负，因此可以无截断地构造GEMM物理写预算。每计划请求节点模式为
360 measured、238未验证插值、190小M的M12锚点，32copy到真实4copy迁移仍未验证。

接入顺序明确为：保留基础成本，单独adapter恢复GEMM物理读写字段；gather提供明确
标定来源及窗口；再做完整计划消融。不能把逻辑gather字节改名为物理字节，或将当前
gather pair的queue系数不经验证直接用于所有GEMM接收者。当前审计不声称完整模型已可用。
容量消融仍沿同一exec24787运行，没有重启。

## GEMM物理预算适配器接入（gather尚未补齐）

M类Lab新增dynamic_resource_candidate/physical_adapter.py::physical_gemm，读取保留的
raw_request_nodes并生成dram_read_kib、dram_write_kib、dram_service_us。prefix取前缀差，
uniform按原基础成本比例分配；逐片段对齐原read分配并检查全阶段守恒。负增量、非有限数、
节点不匹配、成本分解错误直接拒绝，不截断；deepcopy保留输入、service、deps、CPU等。
调用方必须显式传入window_fraction，结果记录其未验证属性；零请求窗口为0。

未触碰gather字段，未擅自选择旧/新response。物理solver运行前还需gather provider和
兼容响应配置，不能将这些中间输入当完整模型。CodeGraph未索引临时模块，直接读取
adapter/replay调用路径。沿用impact-analysis/test-selector/code-review-gate与Python规则。

7项定向测试通过（.14s），验证写恢复/成本不变/输入不变、负增量拒绝、节点错配及
4类非法窗口；Ruff通过。另实际转换anchor/control_forward的uniform/prefix四组输入，
共896expert、3152片段，全部读写守恒且service/deps/CPU保持；写总量约42.933/42.928MiB。
产物physical_gemm_inputs包含源SHA和audit.json，显式window_fraction=1，仅作为消融假设。
未修改生产接口/kernel/default，未做新硬件测量；完整计划与搜索验收仍未完成。
容量消融继续等待同一session24787的跨LLC结果。

## Gather需求来源汇总与迁移缺口

新增dynamic_resource_candidate/gather_request_sources.json，48profile/1488观测，保持
三套协议分离：8T steady16profile覆盖M12/48/529/1205和IO1/1、32/32；16T steady
20profile覆盖M96/180/1718及已有IO状态；16T single-call12profile覆盖M180/1718，
保留同调用PMU成本/预算及准备位置、offset。未合并不同协议中心值、未拟合默认曲线。

8T逐window预算按原session的median idle rate校正后除该window吞吐；raw值一并保留，
不是独立中位数之比，也不是精确单次调用预算。复核原analyze.py后补上该idle校正，
未覆盖历史测量。16T single-call不加入另一次control调用成本，保留同调用关联。
所有profile均31观测，记录源/提取文件SHA，负值保留不裁剪。

现有数据不能直接提供所有真实4copy、小M和宽度的gather物理预算。这是下一批真实
路径采集需补的范围，不允许逻辑字节改名替代；暂不把这份来源表注册成已验证provider。

## 容量消融最终完成与gather生命周期更正

exec24787正常exit0，四条件、四arm、共15376场景，elapsed879.577s。旧total170
与已有finalist逐样本最大差0；read292、read295、no-cap在所有条件也逐样本完全相同。
无PMU条件均值位置MAE第一/第二场2.921/3.566us，中位数位置MAE13.112/11.683us，
四arm相同。结论是当前双team数据不能辨识容量；不能据此删除高并发计划的容量约束，
也不能用此结果宣布queue对未见GEMM/计划成立。decision.json保存完整结论。

重新核对真实runner发现前述“四份缓冲状态/真实4copy gather”措辞不准确：weight_copies=4
只指packed weights。hidden是固定BF16张量；FP32 route-output workspace固定预触碰，
它并不是gather写入的packed-A缓冲。native从resident_scheduled_scratch_pool租借scratch，
gather写入scratch.packed_a。runner每轮触碰216MiB scrub，亦不同于32copy连续lead-in。

gather_lifecycle_audit.json记录源码位置和上述更正。下一采集需固定真实input并保留
每scratch unit的归属、route前序访问与中间GEMM历史；不能用IO4/4冒充真实路径。
旧需求表保留，只作独立协议来源，不据错误的“四份”假设新增硬件实验。本轮没有
新远端采集，完整计划与等预算搜索目标继续，默认不变。

## 真实scratch归属与前序映射

核对冻结native ensure_scratch_config、ResidentScheduledScratchPool和run_async_task：
key为(core_begin,threads)，同key复用packed_a及barrier；resident池跨调用保留，max_rows
保留此前较大分配。因此first-in-plan不等于cold，FP32 route-output workspace不是packed_a。

对anchor/control_forward完整DAG作拓扑与祖先检查，同scratch相邻expert必须具有依赖
祖先关系才定义前序，歧义直接拒绝。scratch_history_map.json保存源SHA和每expert映射。
两计划均9unit；95个M<12中91/93个复用同scratch前序，4/2个为本计划该unit首任务。
前序M>=48直接到小M在这两计划中为0，不能用“大M后接小M”替代真实主要转换。

gather_history_targets.json按宽度与first/reused、小/中/大M分层，仅按计划M中位位置
选目标，不读取耗时择优。后续需真实target/predecessor routes和前序完整W13/W2，
保留跨调用分配状态；映射仅证明所有权和顺序，不等于缓存冷热标签。当前未新增硬件测量，
物理gather provider和完整计划/等预算搜索仍未完成。

## 实际route与同scratch完整前序提取

gather_actual_history/prepare.py读取真实request016/case017文件layer4，SHA核对
afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431，2048x6共12288
flat route全部无重叠且无遗漏。沿用runner的layer索引与int/contiguous转换，对每目标
scratch链验证每个expert的实际M等于计划M。生成inputs.json与独立routes_data.h，
14个capture保留完整同scratch前序，而非合成单expert route前缀。Ruff通过，实际提取exit0。

输入地址重叠揭示直接前序不足：anchor16T expert99/M20与直接前序token交集0，但与
完整同scratch前序交集19；expert180/M73分别0/58。control16T expert140/M4分别0/4。
这些是地址访问集合事实，不等于cache仍驻留，也不包含其他lane和此前调用的访问。

下一采集runner必须复用固定hidden和scratch，并执行真实前序GEMM；本轮只完成可供
runner使用的route/前序输入，没有声称完成采集或复现全计划缓存状态。单个前序M或
独立copy数量不足以替代输入历史。完整模型/新鲜计划/等预算搜索仍未完成。

## 前序完整expert链执行器：独立构建进行中

gather_actual_history/chain.cpp复用已验证gather_impl.o与JIT W13/W2Direct入口，
实际flat routes、同scratch packed-A/intermediate及前序expert顺序；每前序expert四份
独立地址权重，完整owner stripes、Ntile16，8/16T W13 owner1MiB/512KiB、W2 owner
512KiB/256KiB，全stage8/4MiB。单个smoke执行四copy，最终目标gather逐元素验证，
前序完整输出与常量权重标量公式比较（2%容差仅为smoke，不是生产数值契约）。

input为按token区分的合成BF16，weights为常量，实际route保持；这不是实际随机权重
或全计划状态复现。仅回放同scratch前序，不含其他lane、216MiB scrub或跨调用历史。
gap0目标立即接前序barrier后执行，另允许显式gap对照；尚未接PMU或正式计时采样。

本地shell语法通过，源已同步既定Arm路径。远端kernel SHA
1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629、gather object SHA
5b29ebab205ffa2f3bc4207d0f7512e23290b1593c794b7803b5da61247e78fe与既有实验一致。
进程检查没有发现其他native/measure采集；exec69015正在独立构建。首smoke目标
8T CPU272..279 expert213/M4，前序191/100/21/220，命令参数存smoke_args.json。
未改kernel/production/default，不将构建成功或smoke计时当性能结论。

独立构建exec69015最终exit0。两项数值smoke随后完成：8T CPU272..279 target213/M4、
prefix191/100/21/220（exec20810）；16T CPU240..255 target180/M73、prefix96/230
（exec9306）。两者validated=true，目标gather逐元素准确，所有前序W13/W2输出通过
标量检查。smoke.json/smoke16.json、stderr和build_identity已收回；当前无存活构建或
烟测进程。仅保存最后一次worker计时，没有重复性或性能结论。

线程/barrier和运行器是standalone实现，不声称与生产调度等价。下一步增加同进程
随机配对重复与gap对照，先量化计数等待对历史状态的影响；正式PMU与真实全计划迁移
尚待验证，目标保持完整计划预测与等预算搜索。

## 同进程gap扰动正式对照完成

新增chain_gap.cpp/run_gap.py/analyze_gap.py，保留chain.cpp/native烟测版本。每目标
同进程、同buffer、每round同weight copy，0/200/1000us忙等随机交错；每trial重放
同scratch完整前序，全部输出在计算结束后写出。无PMU、无其他lane或scrub，合成值/
实际route。native差异复核仅涉及重复、gap顺序和计时记录，kernel调用体不变。

exec19220构建完成，exec60548两条件烟测数值/时序通过。分析器2项测试通过（.16s），
验证等待不计入服务、负增量保留、重复记录拒绝；Ruff及shell语法通过。随后exec49130
正式两场seed614621/614622，各14target、5warmup/31round、三gap共1512trial；
23.671/23.674s完成exit0，全部原始文件、协议和构建身份已收回。

预先描述性筛查为abs(median paired mean-worker delta)<=max(1us,5%baseline)。
两场各28项均通过，最大绝对配对中位变化.215/.3825us。不是统计等价、PMU扰动
验收或生产性能结论，第一轮copy状态与跨调用真实分配状态仍不同。gap_decision.json
保留全样本和上述决定；允许继续开发计数对照，不直接采纳新T0。

另做事后team envelope诊断（不替代原筛查）：配对中位变化最大2.23us，来自第二场
control16T M18/gap1000，baseline envelope13.82us、变化-2.23us。全部formal worker
start span最大19.66us；含warmup最大66.82us，记录未删除。忙等对平均worker服务
影响小不等于对team包络无影响，PMU版本需分别验证服务和到达/完成区间。不得用
带等待的team envelope替换真实路径T0。当前无存活实验进程，完整计划/搜索仍待完成。

## 实际前序链PMU协议pilot：小M仍不可辨识

新增chain_pmu.cpp/build_pmu.sh/measure_pmu.py/analyze_pmu.py，保留连续/gap版本。
同进程每round随机mode0无PMU目标、mode1 PMU目标、mode2 PMU idle，三组均执行
同前序完整expert链。全部worker准备完成后ARMED，Python开启32个DDRC读写event，
GO后native固定2ms epoch；DONE后disable、ACK，再数值校验和释放下一链。控制核
避开目标team（目标覆盖240则控制核319，否则240），NUMA3分配不变。

exec51750独立构建完成，exec29978两case单轮smoke数值/计数/时间边界通过。M4
smoke出现-70.82/-31.15KiB读写残差，未作为有效预算。随后只对M4/8T与M73/16T
做两场5warmup/31round，seed614721/614722，exec80738完成exit0。两场原始计数、
时序、idle/control全部收回，未扩大到14target。分析器2项测试通过（.12s），验证
逐event窗口时长扣除和负值保留、缺失事件拒绝，Ruff通过。

| 形状 | 第一/第二场读中位KiB | 读MAD KiB | 第一/第二场写中位KiB |
|---|---:|---:|---:|
|M4/8T|20.869 / 11.547|24.890 / 115.620|1.610 / -.138|
|M73/16T|196.881 / 199.903|16.307 / 15.723|67.344 / 75.485|

M4读负值6/14个，写负值8/18个（各31）；M73读无负值，写第一场1个负值。
同协议PMU-control服务配对中位差M4 -.0638/+.0388us，M73 -.0200/-.0025us，
不能从仪器服务变化小推出预算信号可靠。M4不采用，M73仅中心重复，仍有请求状态与
前序drain归因限制。pmu_pilot_decision.json保留全样本统计，不截零或隐藏负结果。

实际history-to-start中位约2387..2416us，协议额外2ms占主要部分；此前gap对照
只到1ms，因此不宣称此协议等价立即执行。下一步缩短已完成counter-enable后的额外
epoch空等，保留完整时间包围，检查背景残差是否下降后再考虑扩展。所有pilot进程
已终止，当前没有新provider或默认变更，完整计划与等预算搜索仍待完成。

## 去掉2ms epoch的对照完成；转向完整计划敏感度

chain_pmu_short.cpp与原native逐行diff仅epoch从Now()+2000000改为Now()，Python
仍在全部counter开启后才发送GO。原协议、二进制和数据完整保留。exec49203独立
构建及两case烟测通过；exec77463两场5warmup31round，seed614741/614742，
M4/8T与M73/16T采集完成exit0，计数包围/数值检查通过，全部产物已收回。
沿用已测request_budget分析器，measure_pmu_short Ruff和shell语法通过。

| 形状 | 两场读中位KiB | 两场读MAD KiB | 两场写中位KiB |
|---|---:|---:|---:|
|M4/8T|25.843 / 31.093|64.310 / 24.322|.845 / .602|
|M73/16T|201.696 / 190.258|10.831 / 15.451|70.715 / 60.390|

实际history-to-start约388..403us；同协议PMU-control服务配对中位差M4
-.0188/-.0325us，M73+.0731/+.0631us。M4读负值13/6个、写各6个，仍不可作为
可靠逐调用预算。不同native会话间比较不是跨版本随机配对，不声称noise变化唯一
归因epoch。short_epoch_comparison.json保存两协议全部统计，未裁剪或重拟合。

停止用相同单调用协议追加小M样本，下一步做完整计划对未辨识gather需求的敏感度，
再决定必须补测的粒度。已先复核两个历史计划的GEMM物理输入：各788片段，显式L=T0，
最大独立读率86.920GB/s，无片段超过295GB/s读主导参考；physical_input_capacity_audit.json
保留结果。这只是输入可行性，不验证预算来源或小M M12锚点，也不宣称295是瞬时硬上限。
当前无存活采集进程，默认未变；gather需求区间及完整计划/等预算搜索仍需推进。

## 完整计划gather需求敏感度与响应迁移诊断

M类Lab新增gather_sensitivity.py，固定两套224expert计划、T0、GEMM物理读写和L=T0，
295GB/s读参考、旧logical/refill响应关闭。gather仅作显式nuisance场景：逻辑input
cache-line覆盖加destination RFO代理作为read，packed写量作为write；0/1/2倍及
仅M<12置零/加倍。不是已测物理预算或认证上下界。queue0与全局.002分别运行，
全局.002是未验证的gather到GEMM迁移压力测试，不是采用参数。

2项单位/基础成本不变/零请求窗口测试通过（.16s），Ruff通过。exec99193首到达
场景20次完整预测完成。queue0 nominal anchor27770.287us、control28530.167us；
small_zero到small_double跨度29.534/19.124us。queue.002 nominal32452.107/33366.913us，
对应小M跨度37.984/40.130us；相对queue0增加约4.68/4.84ms。不能从一个到达场景
推广为全部计划不敏感，随后启动全31个独立到达场景的主要对照。

gather_sensitivity_repeat.py对queue0的nominal/small_zero/small_double，以及queue.002
的nominal共8arm×31预测，首场景必须精确复现首轮结果，全部预测写出后才读取实测。
exec46618当前运行中，保留逐arrival完整phase events。该批属于已见两计划的诊断，
不是新鲜泛化或等预算搜索。

另直接比较首预测arrival与31实测task envelope的均值，phase_diagnostic.json保留
1792个expert-session-arm的各阶段记录。第二场queue0每expert平均有符号误差，
anchor W13/W2=-16.85/-12.51us，control=-19.27/-13.58us；全局queue.002后为
+105.09/+50.02及+98.73/+48.65us。gather仍平均低估约3.85/4.11us。并行expert误差
之和不能解释成makespan贡献，报告不作该加总归因。

这支持下一步分离gather与GEMM接收响应及可隐藏窗口，而不是继续用小gather单次
计数来修复毫秒级全局响应错误。默认未变，完整目标继续。

全31arrival复核exec46618最终exit0，8arm×31=248完整计划预测完成，首arrival均
精确复现首轮。queue0 nominal anchor27760.959us、control28526.904us；对两场实测
分别低估3.264/3.077%、3.790/4.195%。仅小M gather需求0→2倍的同arrival时间
跨度中位30.338/19.124us、最大34.182/19.177us。这个结论限定该读容量-only候选和
两个已见计划，不是所有新计划的上界。

全局queue.002 nominal为32440.972/33362.660us，对两场实际分别高估13.044/13.263%、
12.519/12.045%。全部四个保留arm都选anchor，不能宣称新的选择收益。summary与
decision.json保存完整误差、每arrival跨度和选择；没有按此结果拟合新参数。

下一步优先把gather与GEMM接收响应分开，固定预算/T0检查各自作用，再处理GEMM
可隐藏访存窗口。此结果明确拒绝将gather系数直接全局推广，也说明本阶段不应继续
用小gather计数精度作为完整模型推进的唯一前置。当前无运行中离线预测进程。

## 接收侧queue分离与固定窗口消融

M类Lab simulate.py新增dram_queue_{gather,w13,w2}_us_per_kib，逐receiver读取；缺省
回退旧dram_queue_us_per_kib，显式0覆盖global。所有来源仍以其他expert实际read+write
请求率聚合，own-team排除、预算/容量约束不变；非零response仍要求separate clocks。
不是新增已拟合系数。6项新测试覆盖三阶段解析响应、零覆盖、全量等价、legacy拒绝，
连同旧测试71passed（.21s），Ruff通过。

receiver_ablation.py固定nominal gather假设、GEMM预算、L=T0、read295，首arrival
逐receiver开启.002；exec9565完成12全计划预测，none/all四个完整结果与历史逐字段
精确相等（包括events、预算和feedback统计）。

| 接收响应开启 | anchor us | control_forward us |
|---|---:|---:|
|none|27770.287|28530.167|
|gather|28028.732|28787.573|
|W13|31101.830|32020.894|
|W2|29084.240|29979.797|
|W13+W2|32314.099|33223.855|
|all|32452.107|33366.913|

gather-only相对第二场31实测的每expert平均有符号gather误差+.397/+.175us；
W13/W2仍低估。W2-only虽然总时间较接近，但W2平均高估60.00/56.75us、W13
仍低估20.88/21.92us，不能依总时间选它。phase_diagnostic.json保留分项，干预效应
不可直接相加作为物理makespan贡献。

随后overlap_ablation.py固定三receiver .002、需求和T0，仅将GEMM的L/T0设为
.35/.5/.75/1，gather保持1；exec91215完成8预测，L/T0=1与all完整结果精确相等。
anchor为28427.964/28163.342/28000.093/32452.107us；control为
29148.912/28882.617/28778.925/33366.913us。没有按真实计划分数选比例或重拟合。

缩短窗口同时增加瞬时请求率并缩短请求活跃区间，非单调结果来自此模型结构，
不是实测证明。特别是大M剩余段目前为粗片段，提前完成整个片段请求不能直接解释
成硬件能提前取完所有未来panel。下一步需分离服务隐藏与请求时序假设，以独立证据
标定GEMM响应；本轮不采用统一比例或用总时间误差抵消通过验收。当前无运行中任务。

## 请求结束时钟审计：短窗口同时改变需求时序

Lab predict新增keyword-only record_dram_timing=False；显式开启才在gather worker和
GEMM segment输出dram_end_us。只记录虚拟请求时钟结束，不改调度/计数，默认输出
不变。新测试先因fixture遗漏既有四个零response键失败，补齐fixture后通过；完整
Lab72passed（.18s），Ruff通过。

request_timing_audit.py对两计划、L/T0=.35/.75/1共6次完整预测启用诊断，剥离新增
字段后与对应已冻结overlap结果逐字段完全一致，exec17485正常结束。

| 计划/比例 | 请求早于片段结束数量（共788） | 大M后续段请求结束后的时间占比中位 | 最大无请求尾段us |
|---|---:|---:|---:|
|anchor/.35|669|59.91%|8478.626|
|control/.35|724|60.12%|8534.897|
|anchor/.75|559|12.54%|1750.338|
|control/.75|546|13.00%|1872.271|
|两计划/1|0|0%|0|

“大M后续段”指M>=96、累计节点超过48的片段，每计划54个。当前预算Q按Q/L
发出，memory clock结束即停止pressure，而baseline work可继续。因此缩小L不仅
增加可隐藏时间，还提高初始请求率并清空后半段压力，粗片段允许更早完成未来部分
请求。该表是模型内部诊断，不是硬件prefetch/实际DRAM时间线，也不能证明所有
实际请求必然均匀；它明确否定把该参数仅称为“隐藏比例”的解释。

下一步保持请求时间分布不变，单独检查接收侧服务响应强度；之后用独立阶段竞争
数据标定，不能把短窗口总时间接近当成采用依据。无硬件新采集，默认模型未变，
当前全部离线进程已结束，完整泛化与搜索目标继续。

## 固定请求窗口的阶段损失标定（进行中）

fit_receiver_strength.py保持L=T0、nominal nuisance gather、read295、gather k=.002，
仅搜索W13/W2接收系数。训练只读取anchor session1的31call task envelope阶段均值，
448个W13/W2绝对us残差等权；不以makespan作训练损失。首arrival作粗标定，参数
冻结后才跑两计划各31arrival并评分，数据此前已见，不称新鲜holdout。Ruff通过，
预测器与预算接口不改。

粗网格各{0,.00025,.0005,.001,.002}共25点，最优k13=k2=0，训练MAE20.3879us。
frozen.json已落盘；exec27162正在完成31arrival回顾验证，不能将边界最优解释为
不存在更小的有效响应。

因此仅依据训练边界细化{0,.000025,.00005,.0001,.00015,.0002}的36点，
fit_receiver_strength_fine.py复用相同训练目标，只写训练结果，不读取验证集、不做
新总时间选择。exec8463运行中。早期小正值已改善训练loss，因此粗零点并非响应
不存在的证据；完整细网格结束和冻结前不选参数。两进程为本地离线计算，没有新
硬件采集或默认变更，完整模型泛化与等预算搜索仍待完成。

粗网格及其31arrival验证exec27162最终完成exit0：k13=k2=0、gather k=.002，
anchor中位28017.473us，control28783.352us；两场实际误差分别-2.370/-2.181%、
-2.926/-3.334%。这是训练后冻结的回顾对照，不按总时间重新选参数。

训练集细网格exec8463完成36点，选k13=.000025、k2=.00005，阶段MAE19.9929us，
相对粗零点20.3879us只改善.3950us（约1.94%）。frozen.json保存全部曲线，未声称
统计显著性或泛化成立。系数差异不能直接解释成计算/访存占比，因为需求先验仍有
未验证部分。

evaluate_receiver_strength_fine.py只读取已冻结selected参数，不重拟合，exec83759
正在两计划各31arrival回顾评分。参数已在评价前落盘；未来真实未见计划/等预算搜索
仍为最终门槛，默认模型未更换。

## 固定接收系数回顾完成；新顺序留出已准备

exec83759完成两计划各31arrival评分，参数未再调整。fine anchor28094.345us，
两场误差-2.102/-1.913%；control28861.951us，误差-2.660/-3.070%。仍系统低估，
不能宣布泛化完成。shape_groups.json进一步按第二场width/M分组：8T M<12 W13
平均低估36.10/32.88us，16T M>=49 W13平均高估44.42/45.03us（各5expert）。
这是联合阶段残差，不是独立T0因果诊断，不能用统一正系数同时修复两组。

prepare_receiver_holdout.py复用已验证lane reorder/resource检查，uniformish route
request022/case023 layer20、234expert、maxM697、10x8T，lane成员/窗口/依赖约束保持，
early merge关闭。两个新order seed615101/615102均不同于已有anchor/reverse/small_first/
staggered；加anchor控制共3计划。所有702expert的物理预算转换与T0检查通过，
frontier/counts/jobs/冻结参数存receiver_order_holdout。最初Ruff发现unused Path，已移除，
脚本最终Ruff及run.sh语法通过，未改生产或默认planner。

freeze_receiver_holdout.py对3plan×2model×31arrival冻结预测，candidate使用fine参数，
capacity_control使用原模型和预算；控制anchor逐arrival与原events/完成时间精确比对。
exec70327正在运行，所有预测完成前不启动硬件采集。run.sh已准备、未执行，强制
检查predictions/frozen.json存在；后续两场seed615111/615112，5warmup31round、
四份权重、216MiB scrub、固定预触碰workspace、NUMA3 CPU240..319。

这是已见route上的未测执行顺序留出，不是新route/M/混合宽度验收，也不是等预算
完整搜索；本批不允许按holdout结果修改系数。完整目标继续。

## 新执行顺序留出实测完成：达到本批门槛，但未优于capacity对照

exec70327六组31arrival预测全部冻结，prediction_identity.json记录参数/输入/预测
身份。远端扩展dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f、
workspace cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41一致，
无冲突采集进程。exec89818执行receiver_order_holdout/run.sh两场seed615111/615112
完成exit0，两场数值/trace校验通过，原始trace、session和compact全部收回。
首次本地评分在回传未完成时因缺compact退出，未生成评分；回传完成后用原冻结数据
正常评分，没有重启硬件或改参数。

| 计划 | 新候选预测us | capacity对照预测us | 第一/第二场实测us |
|---|---:|---:|---:|
|anchor|27311.783|27852.100|27644.740 / 27812.770|
|order615101|26637.092|26905.840|27221.050 / 27471.310|
|order615102|26730.552|27051.273|27343.360 / 27616.670|

6个plan/session点：candidate MAPE2.2729%、P90 abs3.2086%；capacity MAPE1.2039%、
P90 abs2.0584%。仅两新顺序4点的MAPE为2.6579% vs1.5830%，新候选仍较差。两模型
两场均选order615101，候选集合regret均0，lane诊断均通过。符合测前MAPE<=3、
P90<=5、regret median<=2/P90<=5的本批描述性门槛，不证明总体分位数或所有宽度泛化。

配对轮次中order615101相对anchor优势中位430.22/410.35us、MAD95.56/206.29us，
两场均29/31轮更快；相对order615102优势117.90/176.73us、MAD104.41/235.11us，
23/20轮更快。不是每轮稳定最优或统计显著性声明。evaluation.json与
paired_order_diagnostic.json保留全部结果，评分器Ruff通过。

结论：新候选通过这批未测顺序检查，但准确度未超过capacity对照，选择无额外收益；
不替换默认，不用此留出回调参数。下一步在冻结参数下扩展混合宽度，然后再推进
等预算完整搜索。所有采集、预测与回传进程已结束。

## 混合宽度留出准备与首预测检查

prepare_receiver_mixed.py沿用冻结Baseline和实际source token indices，按M降序，
在固定不重叠team集合上以无竞争预计完成时间作list scheduling，不读取联合实测或
新候选分数选布局。每LLC40核：mixed_narrow为[4,2,1,1]x5，mixed_wide为
[16,8,8,4,2,1,1]；两域镜像，共80核。PlanV2各宽度/allowed/ragged依赖同步重建，
validate_resources通过，source route/count一致，原anchor控制保留。

mixed_narrow有4T87、2T65、1T82个expert；mixed_wide有16T78、8T94、4T26、
2T16、1T20个expert，各234。实际源token列表传入16T source-conditioned gather，
未用M或总tokens数替代。所有预算转换通过非负/守恒/基础成本检查，独立read最大
分别39.361/86.920GB/s，无超过295读参考的输入，仍不证明预算或小M锚点有效。

predict_receiver_mixed.py --first-only（exec62403）完成，anchor27310.518us与旧
同arrival全结果精确一致，mixed_narrow30431.790us、mixed_wide28684.465us。此为
预测，不是实测性能。Ruff和准备器实际验证通过，无新kernel/default更改。

全31arrival预测exec96095已启动，anchor在输入逐字段相同和首结果等价后复用
上一批冻结31arrival，避免重复计算；两混合组合预计离线约7分钟。参数不变，待
predictions/frozen.json完成后才允许两场实测。run.sh已准备、尚未运行，seed615211/
615212，沿用既定测量协议和门槛。这是混合组合留出，不是等预算完整搜索。

## 2026-09-14：混合宽度预测冻结与实测启动

exec96095最终exit0，mixed_narrow31arrival中位30422.909us、耗时269.305s，
mixed_wide28670.591us、145.571s；anchor复用冻结27311.783us。三组均31arrival，
预测/参数/输入identity测前保存后同步。原扩展和workspace SHA复核一致，无冲突采集。

exec86706已启动receiver_mixed_holdout/run.sh，第一场数值/trace校验通过，第二场
仍运行。模型不修改；预计最优为anchor，后续按实测检验。

评分器evaluate_receiver_holdout.py新增--root及单模型格式支持，指标公式不变。
用既有真实order-holdout数据的多模型/单模型两种schema复核，records/selections/
summary数值均与已保存结果完全相同；Ruff通过。不是重测硬件或调整验收指标。

测前gate仍为MAPE3%、P90abs5%、regret median2%/P90 5%，lane诊断max(100us,10%)。
此为已见route上的新混合组合，不能当作新route或完整搜索验收。

## 混合宽度留出失败：窄team GEMM联合减速明显低估

exec86706两场完成exit0，数值和trace校验均通过。compact/原始trace/日志已分开
收回，评分前核对冻结输入identity；当前无存活采集或回传进程。

| 计划 | 预测us | 第一场实测us/误差 | 第二场实测us/误差 |
|---|---:|---:|---:|
|anchor|27311.783|27961.310 / -2.323%|27995.300 / -2.442%|
|mixed_narrow|30422.909|35611.450 / -14.570%|35594.060 / -14.528%|
|mixed_wide|28670.591|32395.670 / -11.499%|32331.840 / -11.324%|

6点MAPE9.4475%、P90abs14.5699%，完成时间门槛失败。每场mixed_narrow有37个
team不满足lane诊断，mixed_wide有6个。模型两场仍选实测最快anchor，候选集合
regret0；这不抵消时间预测失败。保持原冻结结果，不按本批误差调整参数。

phase_diagnostic.json按width/stage比较31预测与31实测均值，第二场mixed_narrow
1T W13/W2平均低估1241.39/520.20us，2T为666.97/289.37us，4T为315.26/42.13us；
gather平均低估约18–25us，主要差额在GEMM。mixed_wide的1/2/4T也有同方向低估，
16T W13平均仅高估3.07us。所有实际gather→W13边界最小gap为正（约.15us），
未发现这两个阶段的envelope重叠被重复计入。阶段误差不能加总成makespan贡献，
也不能直接证明独立T0错误或唯一归因LLC/DRAM。

此结果否定当前常量receiver系数的跨宽度可靠性。下一步做同route/M/核位置的
无竞争与保留前序的无竞争对照，拆开窄team基线与竞争影响，不能补任意宽度惩罚。
baseline_contrast_targets.json按每width组的M中位位置选择8个目标，不按最大误差
挑点；覆盖narrow1/2/4T与wide1/2/4/8/16T，实际M14..22。当前只准备目标，未启动
新实验。新模型不采用，可靠泛化和等预算完整搜索目标仍未完成。

## 2026-09-14：精确形状基线对照已采完，回传暂受连接故障影响

prepare_mixed_baseline_contrast.py复用isolate_bridge，direct将目标移到首位；prefix
依次提升原同team前序，再执行目标。全部原M/width/core/routes保留，所有非前序任务
静态验证为目标后继；原3联合计划作为批内控制，每模式8隔离+3控制=11计划。
9项既有隔离测试通过（.17s），资源与前序祖先检查通过，Ruff/shell语法通过。

exec10637执行run.sh，direct两场seed615311/615312、prefix两场615321/615322，
均5warmup31formal。四场都输出validated，最终COMPLETE/exit0；数值和隔离trace
检查通过，worker-details已启用。没有重跑或调整模型参数。

回传compact的首次rsync、15秒连接超时重试及原始trace回传均因内网SSH连接超时
退出255；文档中的同机备用Arm-codex在当前环境无法解析。远端实验已经正常结束，
故不重启实验；目前尚未取得这四场本地数据，不能报告基线/竞争分解结果。

analyze_mixed_baseline_contrast.py准备完成：GEMM比较阶段包络，gather比较mean-worker
服务，批内joint-isolated保持配对。2项分析测试通过（.20s），验证不混入gather到达
跨度并拒绝重复pair；不同direct/prefix批次不直接当成同一配对实验。

另准备大M补充：按各width组M的nearest-rank P90选8点，M99..240而不是按最大误差
挑点。prepare脚本新增targets/output/direct-only参数，upper直接隔离11计划的所有
资源/依赖检查通过，目录receiver_mixed_upper_contrast，仅准备、未启动。五个目标
本来就是team首任务，没有独立的same-prefix条件；其余前序对照待当前结果分析后再定。
这些准备不替代当前数据读取。连接恢复后先取回已完成四场数据，再推进基线/竞争诊断。

## 2026-09-14：跨批次控制检查完成，数据回传持续阻塞

分析器补充cross_batch诊断：按session/source_plan/expert/M/width/stage匹配direct与
prefix，分别报告隔离时间变化、原联合控制的批次漂移及二者之差。该差值仅是代数
诊断，依赖未被证明的共同加性漂移假设，不视作前序历史的配对因果效应。缺少模式
显式列为unmatched，重复记录拒绝；direct-only设计仅读取direct目录。

验证命令为.venv/bin/ruff check，对dynamic_resource_candidate下的
analyze_mixed_baseline_contrast.py和test_mixed_baseline_analysis.py检查通过；
.venv/bin/pytest -q tmp/joint_cost_model_20260911/dynamic_resource_candidate/test_mixed_baseline_analysis.py
结果4 passed in 0.20s。新增覆盖共同漂移分离、缺失模式与重复记录拒绝。
这些是分析正确性检查，没有新增硬件性能结果。

本轮仅重试已有四场实验的JSON回传，rsync退出255，SSH返回
“channel 0: open failed: connect failed: Connection refused / stdio forwarding failed”。
与此前超时属于同一Arm连接不可用障碍，已连续三轮阻塞；当前没有存活采集任务，
实验完成状态仍以exec10637的COMPLETE/exit0为准，不重启。四场实测尚未取回，
因此基线/历史/竞争归因待定。可独立完成的本地诊断准备已结束，目标暂记blocked，
须恢复既定Arm连接后取回数据再继续；这不是权限审核拒绝，也不是模型验收通过。

## 2026-09-14：连接恢复，精确形状对照支持优先修竞争响应

JSON回传exec81144和完整目录回传exec42303均exit0，四场compact与原始trace已收回。
analyze_mixed_baseline_contrast.py运行exit0，核对四场isolation_validated、bitwise_correctness、
frontier SHA及每目标31唯一pair，输出analysis.json；未拟合或改动冻结参数。

M14..22的1/2/4T共6目标，W13/W2无竞争基线误差：direct两场MAPE为0.703%/0.948%，
范围分别[-0.255%,2.302%]/[-0.055%,3.152%]；prefix两场MAPE1.552%/1.267%，
范围[0.064%,4.162%]/[-1.051%,3.877%]。这不是所有点满足±3%门槛的声明。

第二场direct的mixed_narrow示例（均值us，增量列为31配对差的中位数）：

|width/M/stage|冻结T0|隔离实测|联合实测|配对联合增量|
|---|---:|---:|---:|---:|
|1T/M16/W13|1680.33|1660.02|4007.75|2298.70|
|1T/M16/W2|841.68|835.44|1923.99|1033.32|
|2T/M18/W13|926.48|914.65|2111.22|1171.68|
|2T/M18/W2|470.83|462.43|1016.60|503.87|
|4T/M20/W13|543.30|543.60|1045.02|500.12|
|4T/M20/W2|279.27|278.58|498.41|220.65|

两场prefix中相应W13配对增量仍约2261–2279、1153–1173、501–510us；
前序隔离没有产生足以解释上述大减速的基线变化。当前已测窄team中位M点支持优先
修竞争响应，不支持用任意宽度惩罚或大幅抬高T0。联合减速包含cache/input/frequency
等环境变化，仍不能唯一归因DRAM/LLC；也不能外推较大M已排除基线问题。

frozen_prediction_diagnostic.json补充原冻结31-arrival预测的描述性比较：1T/M16 W13
预测联合1839.11us，相对冻结T0仅增加158.78us；W2预测1068.30us，仅增加226.62us。
这些预测来自原holdout arrival，当前对照是新批次，不当作配对预测误差或新holdout评分。

已启动预先准备的M99..240直接隔离补充：8目标+原3联合控制，每场11计划，
seeds615331/615332；同route/NUMA3/CPU240..319、5warmup31formal、4weightcopies、
216MiB scrub、固定pretouched workspace、earlymergeoff。扩展/workspace SHA与原记录一致，
预检未发现冲突采集。run.sh仅复用原协议改root/direct-only/seed，bash -n通过。
远端exec96663当前运行，尚无大M结果；未修改生产代码、默认planner或模型参数。

## 2026-09-14：大M精确形状对照两场完成，基线偏差不足以解释联合减速

exec96663两场均validated，最终COMPLETE/exit0。metadata回传exec35797、全量原始
trace回传exec76146均exit0。分析命令：
`.venv/bin/python tmp/joint_cost_model_20260911/dynamic_resource_candidate/analyze_mixed_baseline_contrast.py --root tmp/joint_cost_model_20260911/receiver_mixed_upper_contrast`
输出analysis.json，exit0；两场数值/隔离身份、frontier SHA、31唯一pair检查通过。
direct-only的48条跨模式unmatched为设计预期，不补造prefix结果。

固定原T0，8个M99..240目标×W13/W2的基线MAPE两场0.634%/0.652%，有符号误差范围
[-0.146%,1.862%]/[-0.873%,1.687%]。本批所有16个GEMM点两场都在±3%内，
但覆盖仍是每width按P90 M选择的一点，不能宣称所有M/历史都已经验证。

|计划/width/M/stage|第一场联合减隔离配对中位us|第二场配对中位us|
|---|---:|---:|
|mixed_narrow/1T/M111/W13|2534.12|2509.93|
|mixed_narrow/1T/M111/W2|1510.24|1451.81|
|mixed_narrow/2T/M168/W13|1197.33|1207.57|
|mixed_narrow/2T/M168/W2|822.20|850.95|
|mixed_narrow/4T/M240/W13|882.79|887.93|
|mixed_narrow/4T/M240/W2|56.20|36.95|
|mixed_wide/1T/M99/W13|458.42|469.60|
|mixed_wide/1T/M99/W2|199.35|194.57|
|mixed_wide/8T/M148/W13|7.94|7.77|
|mixed_wide/8T/M148/W2|1.95|0.96|
|mixed_wide/16T/M155/W13|-41.94|-44.80|

小增量需对照MAD：8T/W2两场MAD9.67/8.39us，不能把约1–2us当稳定减速。
16T/W13负增量两场MAD9.69/7.84us，但目标原有3个同team前序，direct改变了历史；
不能将其解释为竞争本身带来加速。不同计划目标的M、前序与时间位置均不同，表中
窄/宽计划差异不是单因素因果对照。

中位M与大M两批共同支持继续冻结现有T0，重点诊断联合响应；保持双GEMM、状态需求
与动态阶段重叠，不添加固定宽度惩罚。下一步要区分全局DRAM压力与同LLC局部压力、
并验证前台阶段/形状的敏感度。已有gemm_pair_response仅4T/M48前台、单后台team、
稳态1/32copy，虽通过单后台宽度留出，却明确未通过多team/真实route迁移；不能把
该旧门槛当作当前混合计划的竞争响应已校准。需先用受控局部/跨LLC多team对照定位，
再决定响应形式和独立校准数据；不直接对失败的holdout残差回归。原holdout失败记录
保留，完整搜索和可靠泛化目标仍未完成。当前无存活采集或回传任务。

## 2026-09-14：1T前台局部/跨LLC多team对照准备与烟测修正

复查gemm_multiteam_response旧记录：4T/M48已有局部/跨域1/2/4/8team、1/32copy
实验，冻结饱和响应count4留出甚至不优于零减速，因此不重复或视作当前问题已解决。
新receiver_locality_response固定1T/CPU319，前台M16/M112、W13/W2、4copy；后台
4T/M48/W13、2/8team、1/32copy，local核尾316、cross核尾276（平移40核）。
36条件，每轮随机顺序；200ms窗口、5warmup31formal、两场seed615411/615412，NUMA3。
首轮只测无PMU服务响应，mean service为主，保留median/p90/p99/max和配对负差。
稳态重复权重与真实route的有限阶段不同，不能替代T0或作为等带宽物理因果校准。

E类Lab探针和Python协议分析，回滚边界为独立receiver_locality_response目录；
无生产API/默认/全局构建变化。复用已读impact-analysis/test-selector/code-review-gate
流程及语言规则。Ruff通过，test_protocol.py两项测试通过（.13s），覆盖工作量/CPU
位置不变式、完整格点、重复/缺失拒绝与负增量保留。shell语法检查通过。

首次1T构建exec42726成功，但smoke exec68144在ARMED前因旧白名单invalid cell退出1，
没有有效性能结果。旧前台只允许M1/12/48/96及1/32copy，后台只允许核尾312/280。
新foreground.cpp仅将白名单限定M16/112、4copy；background.cpp仅改核尾316/276。
逆替换后全文与旧源码相等，计时区/kernel/数值检查源文本不变；不声称二进制等价。
修正版build_v2.sh当前exec55078运行，完成后冻结新身份，重新smoke_v2再决定正式采集。
初次smoke失败文件保留，未覆盖或作为采样使用。初步估计两正式场约12–15分钟，
以成功烟测时间修正。当前未拟合新参数，T0和原holdout结果保持冻结。

build_v2 exec55078成功exit0。前台SHA7a96587000822ab16a220bd80ef3625675d9b8190a64d11d916727c690139f5f，
后台SHA9a7064b3b6fdbb59b1a0f9f990cee78f784a03d06d04274d0f990e2c024c2901，源码hash
与本地一致。smoke_v2 exec98941完成exit0，36cell、32joint，耗时22.747s；回传exec39451
exit0，原始JSONL重新运行分析校验通过、protocol身份匹配。单轮不报告性能结论。
两正式场按72轮线性投影约27.3分钟（含烟测启动开销，非精确时限），修正原12–15分钟
估计；沿用既有Arm实验授权，不新增依赖或默认采用。正式run.sh已启动，等待两场完成。

正式运行handle为exec6046；当前仍运行，尚无正式场结果。

## 2026-09-14：1T局部/跨LLC第一场完成，未复现真实联合大减速

exec6046输出session1 validated，36条件×31formal/5warmup完整，elapsed813.148s；
原脚本继续第二场。回传exec98702 exit0，本地对session1.jsonl重新调用analyze，
检查protocol匹配、31/5、格点/CPU/数值/统计，并与远端analysis做JSON规范化后全量
相等检查，均通过。以下仅第一场描述性结果，不拟合或作为可重复性结论。

8个4T/M48/W13后台、32copy时，前台mean-service配对增量中位（us）：

|前台|同LLC|跨LLC|同round局部减跨域中位/MAD|
|---|---:|---:|---:|
|1T/M16/W13|44.44|27.30|17.54 / 1.17|
|1T/M16/W2|30.72|14.67|16.31 / 2.56|
|1T/M112/W13|93.10|46.20|46.70 / 2.40|
|1T/M112/W2|39.97|27.84|11.98 / 2.28|

1T/M16无背景mean-service为W13 1675.63us、W2 823.56us。增加局部后台造成的
几十us减速远小于真实mixed_narrow M16的W13约2300us、W2约1000us联合增量。
两实验背景数量/形状/阶段/全局活跃核数/调用历史均不相同，不可直接视作同压力
响应误差，也不能将上述差额唯一归因于LLC。M112 W13从2到8个32copy同域后台
增量90.0→93.1us，仍不支持简单按任务数线性缩放。

该证据改变后续优先级：先确认第二场重复，再检查真实双域联合负载与本单域后台
探针之间的压力覆盖差异（本探针最多32后台核+1前台核，真实计划80核），以及后台
形状/阶段和竞争中的请求量是否变化。不能仅凭局部-跨域几十us差拟合局部系数并
宣称修复上千us漏项。现有T0、响应参数、失败holdout记录保持冻结；exec6046仍存活，
第二场未完成，未启动其他硬件负载。

## 2026-09-14：双域压力对照设计冻结，尚未实现或启动

原exec6046仍存活，第二场日志已到round1/31、stderr为空。未启动并行基准。
本地receiver_dual_domain_response/design.json准备20条件：同一1T/M16或112前台、
W13/W2、4copy；后台4T/M48/W13固定32copy，(local,cross)team数为
(0,0)、(8,0)、(0,8)、(4,4)、(8,8)。前3个有背景的32核配置分离总核数不变时
的域分布；最后配置增加到64后台核，保留两种单域控制。核尾316/276，CPU集合
静态检查不重叠且前台319不冲突。最多65活跃核，不能宣称复现80核真实计划。

两场拟用seed615511/615512、5warmup31formal，200ms，预估15–20分钟待新烟测修正。
仅设计，三角色控制器尚未实现/烟测，必须先完成当前第二场并验证重复性再启动。
固定工作量不代表固定实现带宽，仍需防止将请求量变化/全局流量/CPU影响混为
唯一LLC解释；该诊断不直接拟合或采用响应系数。当前模型与原holdout均未改动。

## 2026-09-14：1T位置对照两场完成，转向双域总负载诊断

原exec6046第二场validated32joint、elapsed812.882s，最终COMPLETE/exit0；回传exec40080
exit0。第二场原始JSONL在本地重新运行analyze，protocol/31formal/5warmup检查通过，
与远端analysis做JSON规范化后全量相等。两场均完整，无存活旧采集或回传任务。

8个4T/M48/W13、32copy后台时，mean-service配对增量中位us：

|前台|第一场同域/跨域|第二场同域/跨域|
|---|---:|---:|
|1T/M16/W13|44.44 / 27.30|45.06 / 28.15|
|1T/M16/W2|30.72 / 14.67|31.17 / 13.61|
|1T/M112/W13|93.10 / 46.20|104.90 / 51.20|
|1T/M112/W2|39.97 / 27.84|41.55 / 25.20|

M16主要现象可重复，M112存在跨场变化（如同域W13相差11.8us，超过各场MAD），
不能声称所有条件精确稳定。但两场均仅几十至约百us，未复现真实混合计划的上千us
联合增量。故不拟合新局部惩罚，继续冻结T0和原候选参数，执行已准备的双域诊断。

receiver_dual_domain_response/measure.py复用原进程协议，改为fg/bg_local/bg_cross三角色；
20条件，最大65活跃核，复用同一前后台binary身份，不重编译。analyze.py保留每round
mean-service差，并计算固定32后台核位置对照、64后台核增量及
T_both - T_local - T_cross + T_solo的有符号非加性诊断；该量不是唯一硬件归因。
两项test_protocol.py通过（.17s），Ruff/shell语法通过；覆盖CPU隔离/工作量、非加性
代数、负差、缺失和重复格点拒绝。Python Lab控制和分析变化，不改生产/default。
同步exec76975 exit0，烟测exec75055已启动，1round/0warmup/seed615501；正式场尚未启动。

双域smoke exec75055 exit0，20cell/16joint，13.523s；回传exec90866 exit0，本地原始
JSONL重新分析通过，protocol匹配、与远端analysis规范化全量相等。两场72轮投影
16.23分钟。正式run.sh已启动，两场seed615511/615512、5warmup31formal，先前两场
已结束，无并行基准。烟测不作性能结论，后续完整结果保留原有解释限制。
正式双域运行handle为exec49832，已核验仍存活；当前尚无正式场结果。

## 2026-09-14：双域第一场完成，高总负载出现显著非加性减速

exec49832输出session1 validated，elapsed476.478s，原脚本继续第二场。回传exec58888
exit0；本地analyze重新验证原始JSONL、protocol、31formal/5warmup、格点/CPU/数值，
并与远端analysis规范化全量相等。以下为第一场结果，未拟合，不宣称重复性已验证。

后台固定4T/M48/W13、32copy，前台4copy的mean-service配对增量中位us：

|1T前台|32后台核同域|32后台核跨域|32后台核两域均分|64后台核两域均分|
|---|---:|---:|---:|---:|
|M16/W13|46.00|28.06|34.58|434.40|
|M16/W2|31.33|14.00|27.41|263.58|
|M112/W13|99.90|47.90|94.50|1396.50|
|M112/W2|43.41|28.19|41.02|705.47|

直接逐round计算T_both-T_local-T_cross+T_solo的中位/MAD分别为
M16/W13 360.65/3.35us，M16/W2 221.28/5.27us，M112/W13 1245.80/10.10us，
M112/W2 635.49/16.35us；不是把各条件中位数相加减。固定32后台核分域未出现
同量级跳升，增加总负载到64后台核后出现强非加性。

该结果支持补齐高总负载响应，不能用单域低压力探针校准后直接外推。M16/W13
434us仍小于真实mixed_narrow约2300us，背景形状/阶段及最多65活跃核的覆盖限制
仍在；不能声称真实残差全部解释。没有PMU，不能确认DRAM容量、请求预算变化或
CPU等全局效应的唯一原因。第二场继续验证；再决定高负载下的独立请求/服务计数
诊断，不用本场残差直接拟合常数。当前T0、原候选参数和失败holdout结果不变。

## 2026-09-14：双域两场完成，高负载非加性现象可重复

exec49832第二场validated16joint，elapsed476.549s，最终COMPLETE/exit0。回传exec21734
exit0；本地重新分析session2.jsonl，protocol、31formal/5warmup、格点/CPU/数值与
远端analysis规范化全量相等检查通过。当前无存活采集或回传任务。

64后台核下mean-service配对增量中位us（第一/第二场）：
M16/W13 434.40/436.32，M16/W2 263.58/262.40，M112/W13 1396.50/1390.40，
M112/W2 705.47/686.53。第二场相应MAD2.85/2.96/14.00/12.92us。
逐round非加性中位第一/第二场分别360.65/361.13、221.28/213.96、
1245.80/1236.50、635.49/617.25us。支持高总负载非加性现象可重复，不代表
每点无漂移，也不意味着全真实计划残差已解释或候选模型已通过验收。

下一步receiver_dual_domain_pmu/design.json仅设计，未实现/启动：复用相同binary，
原20前台条件+4非空BG-only位置+1idle，每round含PMU与无PMU控制；500ms窗口
降低M112单调用计数分辨率影响，5warmup31formal、两场seed615611/615612，预计
35–45分钟待烟测修正。检查原始计数、running>=.99、单调用比例<=.05、PMU相对
无PMU mean-service偏差<=5%，数值/CPU/完整格点均需通过。

现有gemm_multiteam_response/events.py的core覆盖248..311及316..319，会漏掉当前
244..247和312..315，不能原样复用。新设计明确采集244..275、284..315、319共65核
的L2 access/refill/writeback，加16个NUMA3 DDRC的read/write/command/occupancy，
共259事件。DDRC范围与独立容量探针一致；32个read/write事件不是32个控制器。
聚合DDRC不能拆成唯一前台DRAM量，foreground refill才可按前台CPU独立观察；
occupancy/command不当作前台访存延迟。本设计无cycles/instructions，不推断频率。
T0及候选参数继续冻结，不按任务数或核数拟合本次跳升。

## 2026-09-14：高负载PMU配对控制实现与烟测启动

receiver_dual_domain_pmu实现50条件（25基础格点各PMU/无PMU）：原20前台条件，
4个BG-only和1idle，500ms；复用同前后台binary，控制器CPU240。PMU覆盖65核的
3个L2事件与16DDRC的4事件，共259事件。RESET只清计数，不清累计enabled/running
时间，分析使用累计时钟差。所有active角色ARMED后启用，GO后等待全部DONE再
停计数；保留原始count/时间及原native结果，计数包络与完整调用边界差异仍需审计。

E类Lab Python采集/分析，无生产/default变化。复用已读impact-analysis、test-selector、
code-review-gate规则；linux_perf_event.py复制旧实现不改。三项test_protocol.py通过
（.17s），Ruff/shell语法通过：覆盖新增8核、259事件、50唯一格点/成对命令/CPU隔离，
RESET累计时间处理，空载扣除负值保留，缺失/重复拒绝。未用单元测试宣称PMU准确。

analyze.py检查每team非空、单调用比例<=5%、counter running>=99%，并按同round
前台mean-service计算PMU相对无PMU的中位百分比（各条件<=5%）。DDRC按每个控制器
实际enabled时间计算有符号idle-corrected流量；core events按角色核集合及完整调用
数计算，不能把联合DDRC总流量当成前台独占DRAM量。BG-only提供独立组请求率。

同步exec84828成功，预检未发现冲突采集；烟测exec92903已启动，seed615601、
1round/0warmup，正式两场尚未启动。通过质量门槛并取回原始数据复核后才运行run.sh。

PMU烟测exec92903成功exit0：50cell，count_resolution/instrumentation均通过，最大
单调用比例0.026316，耗时44.806s；回传exec73278成功。原始JSONL本地重新分析通过，
protocol身份匹配，远端analysis规范化全量相等；前台paired PMU/control偏差最大
0.75748%，单轮仅为仪器烟测，不作性能/物理结论。按两场72轮线性投影53.77分钟，
修正此前35–45分钟估计；既有Arm实验授权范围与无预算约束不变。run.sh正式两场
已启动，seed615611/615612，5warmup31formal，模型参数不变。
正式PMU运行handle为exec43830，已核验仍存活；当前无完整正式场次。


### PMU session1 completed: preliminary resource diagnosis

`receiver_dual_domain_pmu/run.sh` session1 (seed615611, 5 warmup / 31 formal,
500 ms, NUMA3, foreground 1T/M16 or M112/W13 or W2, background 4T/M48/W13,
32 copies per background team) completed in 1603.420 s. Transfer exec57085
exited 0. Local `analyze.analyze` recomputation from session1.jsonl matched the
remote session1.analysis.json in full after JSON normalization; protocol identity
matched protocol.json. Both quality gates passed: maximum single-call fraction
0.026316; maximum absolute paired median PMU/control foreground perturbation
0.176035%. Session2 is running under the original exec43830; no model refit.

Signed idle-corrected background-only median DRAM read rates (GB/s): local8
150.171, cross8 149.499, split4+4 141.232, dual8+8 275.953. These placements
use 32, 32, 32, and 64 background cores respectively. With a foreground, dual8+8
read rates are 276.105–277.889 GB/s. This is near the independently measured
read-dominated 295 GB/s reference, which is not a universal capacity bound.

| Foreground | Local8 delta us | Cross8 delta us | Dual8+8 delta us | Dual FG L2 refill/call change |
| --- | ---: | ---: | ---: | ---: |
| M16/W13 | 45.090 | 28.980 | 434.940 | +0.237% |
| M16/W2 | 32.127 | 14.905 | 265.639 | +0.169% |
| M112/W13 | 99.700 | 55.300 | 1390.900 | -4.965% |
| M112/W2 | 38.300 | 29.240 | 683.640 | -3.133% |

Deltas are medians of within-round PMU-on foreground mean-service differences
against matching isolated M/stage. Refill changes are medians of within-round
ratios, using signed idle-corrected role counts per complete call. The large
dual-domain slowdown is not accompanied by increased foreground L2 refill/call
in this experiment. This motivates examining nonlinear service response under
high shared load; it does not uniquely establish DRAM queueing, identify core
frequency effects, or assign global DDRC bytes to the foreground. Boundary-call
enclosure remains, particularly for M112; small refill changes must not be
interpreted as precise changes in physical demand. Session2 repeatability and
transfer to finite real-plan dynamic overlap remain unverified.


Session1 background-only paired check: dual8+8 versus the matching single-domain
8-team group reduces completed-call throughput by 11.4037% (local) and 11.5605%
(cross), using each role's sum(team_calls)/window_ns. Corresponding signed
idle-corrected L2 refill/call changes are -0.7112% and -0.6407%. Within-round
ratio of dual read GB/s to the sum of the two single-domain read GB/s has median
0.920152. Thus background service also slows without a corresponding refill/call
increase. Throughput and DDRC ratios use different observables and must not be
forced equal or interpreted as foreground-attributed DRAM budgets. These remain
single-session fixed-stage observations, pending session2 repeatability.


### PMU session2 completed and raw-data parity verified

Session2 (seed615612, same frozen protocol, 5 warmup / 31 formal) has a complete
JSONL terminal record and remote analysis, elapsed1603.333s. Transfer exec71549
exited0. Local recomputation of both sessions matched remote analysis in full,
including protocol identity and rounds/warmup/seeds. Session2 count-resolution
and instrumentation gates passed; maximum single-call fraction0.026316 and
maximum absolute paired median PMU/control perturbation0.176247%.

Dual8+8 foreground increment medians / MAD (us), session1 then session2:
M16/W13 434.940/2.350 -> 437.920/2.970;
M16/W2 265.639/2.169 -> 271.631/2.425;
M112/W13 1390.900/8.900 -> 1416.300/6.700;
M112/W2 683.640/17.550 -> 733.180/16.330.
The M112/W2 increment shifts7.25% across sessions, exceeding either session's
within-round MAD; do not claim every response point is a stable constant.
Session2 corresponding FG refill/call changes are +0.0566%, +0.2050%, -4.7860%,
and -3.7899%. Large slowdown again does not accompany increased FG refill/call.

Session2 BG-only read GB/s: local8 149.596, cross8 150.115, split4+4 141.421,
dual8+8 275.706. Dual versus matching single-domain group throughput changes:
local -11.5933%, cross -11.5660%. Qualitative high-load nonlinearity repeats,
but this is repeatability of the same fixed-stage grid, not unseen-M/team/route
or dynamic-overlap generalization. Current measurements leave a large gap
between ~150 and ~276 GB/s; intermediate high-load points are needed before
selecting a response curve. Model coefficients and production defaults unchanged.
The original exec43830 observation handle still returned live after completed
artifacts were verified; no restart was issued and no terminal exit is claimed.


Intermediate-pressure E-class Lab collector prepared under receiver_pressure_curve.
Only measure.py placement grid, protocol metadata, run path/seeds and test grid
changed from prior collector; analyzer/events/linux_perf_event.py are unchanged.
Four tests passed in0.18s via .venv/bin/pytest -q
 tmp/joint_cost_model_20260911/receiver_pressure_curve/test_protocol.py.
Ruff on collector/analyzer/events/tests and bash -n run.sh passed. Eight placements
produce80 paired cells/round; fit and held-out placements frozen in design.json.
Existing native binaries retained. Smoke and formal target collection pending.


Pressure-curve smoke exec92740 exited0: 80 cells, seed615701, 1/0 rounds/warmup,
75.976977s; count-resolution and instrumentation passed. Transfer exec35476
exited0. Local raw recomputation matched remote analysis in full and exact frozen
protocol identity; maximum absolute paired perturbation1.615144%, maximum
single-call fraction0.026316. Smoke is instrumentation validation only.
Revised two-session72-round projection91.17 minutes (previous85.52 estimate).
Formal run.sh launched under exec22056 within existing Arm experimental scope,
seeds615711/615712, 5 warmup/31 formal each. No coefficient/default changes.


### Intermediate-pressure formal collection completed

Original SSH observation exec22056 terminated255 with transport timeout/broken
pipe after interruption. Fresh remote inspection found both session analysis
artifacts, session2 round31/31, empty stderr, and no matching collector/native
processes (pgrep exit1). No restart. Transfer exec30120 exited0. Both raw JSONL
terminal records, frozen protocol, seeds615711/615712 and5/31 counts verified;
local analyze.analyze results matched remote analysis in full after normalization.
Elapsed2723.697/2723.888s. Both count-resolution/instrumentation gates passed;
max single-call fraction0.026316 both, max paired perturbation0.301825/0.315857%.

BG-only median read GB/s session1/session2: 4+4 141.24/141.27; 5+5
183.81/184.22; 6+6 224.81/225.06; 7+7 258.39/257.97; 8+8 275.79/275.46;
8+4 223.29/223.32; 4+8 224.56/224.16.

Paired PMU-on FG mean-service increment medians (us), order
M16/W13, M16/W2, M112/W13, M112/W2:

| Placement | Session1 | Session2 |
| --- | --- | --- |
| 4+4 | 35.34,26.14,96.10,42.98 | 36.26,26.75,95.10,39.13 |
| 5+5 | 55.99,36.59,145.50,69.16 | 57.26,39.20,141.10,66.55 |
| 6+6 | 108.79,67.81,305.00,159.48 | 110.37,68.62,291.10,155.46 |
| 7+7 | 234.54,139.40,709.20,373.32 | 239.80,141.59,699.60,382.43 |
| 8+8 | 434.03,268.78,1403.60,710.86 | 438.49,270.13,1382.00,718.26 |
| 8+4 | 120.84,77.34,334.80,161.76 | 126.91,80.56,322.10,163.67 |
| 4+8 | 112.28,68.24,322.10,151.60 | 119.08,73.14,355.90,169.41 |

No coefficients fitted in this readout. Fit placements remain4+4/6+6/8+8;
held-out5+5/7+7/8+4/4+8 timing has now been inspected descriptively and must not
be used for parameter or family selection while claiming untouched validation.
Nonlinearity repeats; placement and session differences remain. Independent BG
traffic is a measured feature here, not yet a validated plan-derived prediction.
Full-plan holdout and equal-budget complete-search acceptance remain outstanding.


### Bounded pressure-response shape diagnostic

Command: .venv/bin/python tmp/joint_cost_model_20260911/receiver_pressure_curve/fit_response.py.
Output: receiver_pressure_curve/response_diagnostic.json, retaining source SHA256,
all signed per-cell errors and fitted coefficients. Raw sessions are revalidated
before fitting. Initial execution found a tuple/list comparison mismatch in the
analysis record filter; corrected before any result artifact was written.
Fixed independent read reference C=295GB/s, rho=BG-only read/C. Compare zero,
linear a*rho, and rational a*rho/(1-rho). Fit one nonnegative least-squares
coefficient per M/stage, using only session1 paired increment medians at
4+4/6+6/8+8. Freeze session1 background-only rate features for both sessions.
No optimization of capacity or family selection by holdout. Heldout timings were
previously viewed descriptively: results are bounded diagnostic validation.

Heldout increment MAE us (session1/session2): zero194.541/198.644;
linear165.852/161.749; rational14.483/18.676. Rational maximum absolute error
30.585/44.475us. Rational coefficients us: M16/W13 30.4471, M16/W2 18.8793,
M112/W13 97.6945, M112/W2 49.5277. M16 heldout errors are all negative in both
sessions, leaving systematic underprediction despite the large overall reduction.
Asymmetric placements and session drift remain; do not claim full local-response
identification or no residual bias. Coefficients are per shape, background pressure
is independently measured rather than inferred from plan, and response is steady
fixed-stage only. This is not a full-plan model or evidence of physical queueing.

Focused test test_fit_response.py passed1 test (coefficient recovery, zero-pressure,
monotonicity, negative coefficient clamp, invalid domain); Ruff passed. Current
production model unchanged. Next: examine whether independently measured receiver
features explain coefficients, then validate new receiver shapes and dynamic
pressure rather than introduce a free coefficient for every shape.


### Receiver feature proportional-transfer diagnostic

Output receiver_pressure_curve/receiver_feature_transfer.json uses session1
isolated PMU observations and existing fixed-C rational coefficients. For each
stage, train coefficient only at M16 and transfer to M112 in proportion to solo
time, L2 refill/call, or solo DDRC read-rate * native window / complete calls.
The last quantity is an enclosure proxy, not exact attributed foreground bytes.
No M112 coefficient is used to construct the transferred coefficient; it is used
only to recover the existing unit-pressure predictions for scoring. This is an
exploratory check on descriptively viewed shapes, not prospective generalization.

M112/M16 ratios: W13 time6.8761, refill5.2452, read proxy3.5673 versus fitted
response coefficient3.2087. W2 time6.9020, refill5.2809, read proxy4.7638 versus
response coefficient2.6234. Fixed demand-proportional transfer is insufficient.
Heldout-placement M112 increment MAE us, session1/session2:
W13 time407.911/406.136, refill221.524/219.749, read29.774/32.833;
W2 time300.131/293.576, refill185.259/178.704, read148.615/142.060.
Time/refill transfers systematically overpredict. Read proxy is closer for W13
but still overpredicts W2. Do not adopt a different feature per stage based solely
on these already viewed outcomes, or fit an exponent to two M values and claim
cross-M generalization. Request timing/overlap/history or exposure may explain
sensitivity differences, but these mechanisms remain unseparated. Need new
receiver M/history observations with independently frozen predictions.


### Prospective intermediate-M receiver validation prepared

receiver_m_holdout/design.json freezes36 predictions before measurement, using
per-stage affine coefficient interpolation between M16 and M112 and previous
session1 BG-only pressure at6+6/8+8. New M48/52 and96/100 pairs distinguish
complete-block and four-row-tail shapes; M16/112 are concurrent anchors. Six M,
two stages, three placements plus3 BG-only/idle, paired PMU/control:78 cells.
5warmup31formal, seeds615811/615812 planned. This is an empirical interpolation
null hypothesis, not physical overlap identification or production adoption.

E-class isolated Lab implementation: foreground.cpp expands only M whitelist.
Reverse replacement yields exact original source; no binary-equivalence claim.
Collector grid expanded; analyzer/events/perf helper unchanged. Four protocol
checks passed0.16s, Ruff and build-shell syntax passed. Sync exec10508 exited0;
Arm standalone foreground build launched. Target smoke/formal remain pending.
Frozen predictions and existing production model unchanged.


Receiver-M Arm build exec98501 exited0. Foreground binary SHA256
3f27b4e43affb888c99eda642d6c3c7f5dda473a2cee1a0334c66d50d230c04e;
source758d598751daa0d6c981d0991a8b24c1661a677c523ebac429c72060a45d6dc6
matches local. JIT source1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629.
Protocol binary identity frozen and synced (exec80595 exit0). Smoke exec44458
started, seed615801, one round/no warmup. No formal measurements launched yet.


Receiver-M smoke exec44458 exited0:78 cells, seed615801, 1round/0warmup,
76.456098s. Count-resolution/instrumentation passed, max single-call fraction
0.026316. Transfer exec89147 exited0, raw analyze recomputation matched remote
analysis and exact protocol identity; max absolute paired PMU/control
perturbation0.929516%. Smoke only validates measurement, not interpolation.
Revised formal two-session estimate91.75minutes; formal run.sh launched under
exec27902, seeds615811/615812,5warmup31formal, existing authorization retained.
Frozen prospective predictions unchanged. No production-model change.


### Intermediate-M session1 prospective score

Transfer exec79462 exited0. Local inline Python invoked analyze.analyze on
receiver_m_holdout/session1.jsonl: all gates passed, normalized full analysis
parity, exact frozen protocol and31/5/615811 identity verified. Terminal elapsed
2741.439784s. Frozen design predictions were scored without refitting against
median within-round joint-minus-solo FG mean time (31 paired PMU-on samples).
Across two nonzero placements and both stages, anchor8-cell increment MAE is
8.678702us; unseen-M16-cell MAE is43.162267us. These are response-increment
errors, not complete-plan or independent-T0 absolute errors.

At8+8, predicted/observed increment us:
M48 W13 758.76/599.49, W2 417.62/317.39;
M52 W13 798.98/882.93, W2 435.95/464.58;
M96 W13 1241.37/1203.64, W2 637.57/590.69;
M100 W13 1281.59/1278.20, W2 655.90/681.79.
M48/52 W13 paired-increment MAD3.83/3.38us, W2 7.10/4.55us.
Thus the first-session M48->52 response change is substantially larger than
smooth M interpolation predicts; M96->100 changes differ. This supports further
block/tail/history sensitivity diagnostics, not uniquely identifying a mechanism.
No coefficients changed. Session2 repeatability and anchor drift audit remain
pending; full-plan and equal-budget complete-search acceptance remain open.

Session1 block/tail contrast diagnostic (inline Python, raw reanalysis/parity
passed): calculate within each round [(joint_hi-solo_hi)-(joint_lo-solo_lo)],
then median/MAD; this differs from subtracting separate aggregate medians.
At8+8, M48->52 W13 PMU-on281.960/5.500us, PMU-off283.980us;
W2 on144.170/8.620us, off145.840us. M96->100 W13 on78.330/18.600us,
off101.380us; W2 on93.040/21.390us, off98.490us. At6+6, on paired
median/MAD: M48->52 W13 73.490/13.630us,W2 37.390/5.940us;
M96->100 W13 56.240/16.490us,W2 36.070/9.200us.
The strongest M48->52 contrast also appears without PMU, so PMU collection
alone does not explain it. This is a difference of complete-shape responses,
not a direct isolated-tail latency measurement.

Solo median W13 times M48/52=4923.37/5362.51us and M96/100=
9841.40/10283.30us: the respective differences of medians439.14/441.90us
are similar despite different high-pressure response contrasts. Solo refill/call
increases134460.8 versus158009.4, which also does not explain the substantially
larger first pair by a simple proportional rule. W2 solo times2428.06/2641.52
and4861.58/5073.59us. BG-only read6+6=224.5268,8+8=275.4035GB/s,
close to frozen source224.8114/275.7897GB/s; primary predictions remain frozen.
These single-session diagnostics motivate history-dependent response analysis;
they do not establish causality or dynamic/full-plan generalization.


### Intermediate-M session2 repeatability verified

Transfer exec99529 exited0. Inline Python reanalyzed both raw sessions, confirmed
all quality gates, normalized full remote-analysis parity, exact frozen protocol,
31/5 rounds/warmup and seeds615811/615812. Session2 terminal elapsed2741.865100s.
Frozen predictions unchanged. Increment MAE anchor8 cells8.678702->9.986951us;
unseen-M16 cells43.162267->45.453719us. These are steady-stage response errors,
not full-plan errors. Session2 at8+8 observed increments W13/W2 us:
M48 598.19/331.08; M52 893.73/466.66; M96 1183.15/590.81;
M100 1292.30/669.40. M48 overprediction and M52 underprediction at8+8 repeat.

Within-round difference-of-increments median/MAD at8+8, session1->session2:
M48->52 W13 281.96/5.50 ->294.34/6.26us;
M48->52 W2 144.17/8.62 ->141.79/9.63us;
M96->100 W13 78.33/18.60 ->110.11/37.55us;
M96->100 W2 93.04/21.39 ->79.98/18.23us.
The first-pair excess response is repeatable; the later pair is noisier and its
exact magnitude should not be treated as a fixed constant. Smooth affine-M
receiver interpolation is inadequate for this grid. This does not identify a
unique cache/history mechanism. Next diagnostic must separate block/tail history
response or independently characterize request exposure, without absorbing these
heldout residuals as arbitrary per-M corrections. Production model unchanged;
full-plan holdouts and equal-budget complete-search acceptance remain outstanding.


### Block-response follow-up: source audit and measurement contract

Read-only source audit of receiver_m_holdout/foreground.cpp: CodeGraph could not
uniquely index this ignored file; targeted source inspection followed. Each call
uses full per-thread N stripe, iterates row=0..M in steps12 against the same B
pointer, then advances iteration and selects iteration%4 weight copy on the next
call. ARMED requires at least64 prior completed calls. A arrays are reused, and
W13/W2 run separately. Consequently history is not only the number of preceding
12-row blocks: changing M changes the reuse interval of each of four B copies
across calls. No cold-first-expert interpretation is justified by this protocol.

Next isolated Lab measurement contract (prepared, not implemented or launched):
- Preserve these exact complete-call sequences, M48/52/96/100, W13/W2, CPU319,
  background4T/M48/W13 with32 copies at0/6+6/8+8, NUMA3 and500ms windows.
- Add an optional per-block timing mode, recording every block boundary within
  each complete call, plus outer-call time. Retain uninstrumented paired controls;
  do not replace the entire call with repeated standalone tails.
- Report the sum of prefix-block competition increments, terminal-block increment,
  and outer-minus-sum overhead separately. Compare first4 blocks of M48 and M52,
  first8 of M96 and M100; report the difference-of-shape response as prefix change
  plus added tail, rather than assigning it all to the tail by construction.
- Keep observation selection identical by whole-call completion edge. Preallocate
  timing storage, avoid allocation/output in timed loops, check numerical output,
  CPU placement, count resolution and per-shape instrumentation perturbation.
  An unexplained >5% paired whole-call instrumentation shift fails attribution.
- First use PMU-off timing/control pairs. The already completed PMU data remain
  independent resource evidence; extra per-block PMU reads are not required.
- After this attribution, if prefix changes explain the difference, independently
  vary B-copy reuse interval while retaining within-call block order. If added-tail
  response dominates, vary prefix length with the same tail kernel. Neither branch
  is yet a physical-causality conclusion. New unseen prefix lengths are needed
  for validation after fitting; current four shapes become diagnostic data.

This contract changes no production model or defaults. Full-plan generalization
and equal-budget complete planner search remain required after local attribution.

E-class block-timing foreground implementation prepared in receiver_block_timing:
copy existing foreground and build only into new optional Lab directory. Append
explicit timing0/1 command field; preserve kernels, packed buffers, B rotation,
64-call lead-in, barriers, whole-call completion selection and numerical checks.
Only1T and active W13/W2 accepted. Preallocated records retain11 boundaries for
up to10 blocks (M<=112); timing-off has no boundary clock calls. Report means
per block and outer-minus-block-span overhead on identical completed calls;
validate ordering and enclosure. Native build/numerical and paired perturbation
checks pending. Shell syntax passed. No production defaults or model equations
changed. Shared impact-analysis/test-selector/code-review-gate applied; rollback
boundary is this new Lab directory and its record, retaining prior raw evidence.

Block-timing Arm build exec8950 exited0. First correctness invocation was not
executed because automatic permission review timed out; one permitted retry
exec68669 exited0. Inline remote .venv/bin/python drove numactl --membind=3
foreground_native through24 active cases (M16/48/52/96/100/112 x W13/W2 x
timing0/1),30ms windows and unchanged64-call lead-in. All numerical/CPU checks,
block counts, positive block intervals, nonnegative overhead and block-sum plus
overhead versus outer mean (JSON rounding tolerance2e-5 relative/2ns absolute)
passed. Three invalid inputs (timing2,stage-1,M113) were rejected. These short
checks establish correctness, not performance or count-resolution acceptance.
Transfer exec32904 exited0; local native_correctness.json has all24 cases and
local foreground source hash matches build_identity.txt. Native SHA256
6adf8feea18e28ae19f3945ea2e632d0a7ec97776fc65e33ae3837820bc0f6ee;
source1e25675e0d3d0f120a725196ba1c61a27bad55a5f10fcdf4d871952ee50b3f41;
JITsource1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629.
Next: paired PMU-off timing/control collector and complete-grid analyzer, then
smoke instrumentation gate before formal measurements. Existing production
model and frozen receiver predictions unchanged.

Block-timing PMU-off collector/analyzer prepared. Grid is four diagnostic shapes
M48/52/96/100 x W13/W2 x0/6+6/8+8 placements x timing0/1 =48 cells/round.
Collector retains existing three-process ARMED/GO/DONE/ACK and cleanup, native
identity checks, shuffled rounds and exclusive output creation; PMU code removed.
Protocol fixes500ms windows,5warmup31formal, seeds615911/615912, source-verified
foreground identity and unchanged background binary. Analyzer verifies complete
grid, duplicate/missing observations, active roles/CPU groups, native identities,
finite timings, per-team counts, block cardinality and accounting; instrumentation
and count-resolution limits remain5percent. Tests command .venv/bin/pytest -q
 tmp/joint_cost_model_20260911/receiver_block_timing/test_protocol.py passed8 tests
in0.16s (valid full grid, six corruption cases, perturbation-gate failure).
Ruff on collector/analyzer/tests passed. Remote smoke and formal collection not
yet launched; these local checks do not establish native measurement perturbation.

Block-timing smoke collector completed48 cells (seed615901,1/0 rounds/warmup),
elapsed49.258311s. Exec31152 exited1 in analyzer: background native reports median,
not mean/max. Raw data retained and fetched (exec90336 exit0), no recollection.
Corrected role-specific statistics validation and synthetic fixture to match real
background schema;8 tests passed0.14s and Ruff passed. Local original-raw analysis
passed both gates, max paired timing/control perturbation0.811687percent and max
single-call fraction0.023256. Frozen protocol and1/0/615901 identity verified.
Corrected analyzer synced; remote reanalysis/full parity pending. No coefficients
fit and no block-performance conclusions drawn from smoke. Two-session formal
projection from smoke59.11minutes, within existing experimental authorization.

Remote smoke reanalysis exec80228 exited0; SHA256 matches local analysis exactly
(6d24a2886afbd0897ffc8e1ca0b17854287656b2b0731088ca360835f0eb6ef0),
confirming full byte parity after role-specific correction. Formal run.sh shell
syntax passed, sync exec40995 exited0. Launching two sessions, seeds615911/615912,
5warmup31formal each,48 cells,500ms, PMU-off timing/control. Estimated59.11minutes
from smoke; prior Arm experimental authorization retained. No model refit.


### Block-timing session1: prefix cancellation is measurable

Transfer exec60094 exited0. Inline Python raw analyze passed all gates, full
normalized remote-analysis parity, exact frozen protocol and31/5/615911 identity.
Elapsed1763.627305s; max paired timing/control perturbation0.213289percent,
max single-call fraction0.023256. For each round compute shape-response difference
as prefix-response difference + added-tail response + outer-overhead difference.
Each response subtracts that shape's same-round solo observation. Values below
are medians of the respective paired components (their medians need not add).

At8+8, prefix / added-tail / overhead / whole-shape difference us:
M48->52 W13 -6.560 /292.049 /0.119 /283.550;
M48->52 W2 5.894 /139.815 /0.101 /147.090;
M96->100 W13 -140.540 /244.073 /0.012 /103.970;
M96->100 W2 -18.483 /91.334 /0.005 /74.240.
Corresponding prefix/tail/whole MAD us:
5.010/2.493/7.260;5.603/1.535/6.710;
33.770/2.777/35.040;11.048/6.809/14.380.
At6+6, prefix/tail/whole medians us:
M48->52 W13 1.260/78.209/79.420,W2 -0.192/37.957/38.100;
M96->100 W13 -11.100/61.513/49.330,W2 0.471/36.835/36.710.

M48->52 is primarily added-tail response in this session. For M96->100 W13,
a substantial negative prefix-response difference offsets the positive tail;
the smaller whole-shape difference is not a direct tail-cost estimate. This is
measured decomposition, not proof of the B-copy reuse-interval mechanism. Next:
inspect per-block placement of the prefix change and compare timing0 against
prior uninstrumented complete-shape controls, then session2 repeatability.
No model parameters changed; dynamic and full-plan acceptance remain outstanding.

Session1 per-block follow-up, raw reanalysis and full parity rechecked. At8+8,
M96->100 W13 prefix-response differences by zero-based block0..7 (median us):
-0.98,-8.10,-14.42,-18.89,-24.13,-22.17,-22.71,-21.20;
MAD2.47,4.20,5.70,7.79,7.14,6.29,7.31,7.54us. The difference grows after
block0, so an interpretation limited to a changed first cold-B block is inadequate.
W2 corresponding differences0.79,-1.82,-0.69,-1.71,-3.58,-2.90,-3.92,-4.88us.
M48->52 W13 differences -4.28,-1.09,-0.44,-1.82us; no comparable large prefix
cancellation. Descriptive prior-control audit at8+8, timing0 current versus previous
receiver_m_holdout PMU-off sessions1/2 increment medians us:
M48W13 606.70/600.74/600.66,W2 294.18/320.98/329.20;
M52W13 891.83/883.16/893.36,W2 438.40/465.05/467.03;
M96W13 1139.98/1180.39/1194.09,W2 570.04/595.05/588.13;
M100W13 1291.30/1273.40/1294.40,W2 665.46/684.34/670.16.
These changed-binary/session comparisons are descriptive, not proof of no common
instrumentation bias. Whole-stage relative perturbation can hide uncertainty in
smaller competition increments: within-round timing1-minus-timing0 increment
median/MAD at8+8 is16.82/65.84us forM96W13, -12.70/19.42us forM100W2;
M48/52 each stage medians lie between-0.46 and1.54us, MAD6.39..7.08us.
Retain cancellation as session1 evidence; do not calibrate its exact magnitude
before repetition and common-control drift assessment. Second session remains active.


### Block-timing session2: repeated prefix cancellation

Remote session2 stdout reached31/31, stderr empty; transfer exec58809 exited0.
Original raw session2.jsonl reanalysis passed all gates, full normalized JSON
parity with remote session2.analysis.json, exact protocol and31/5/615912 identity.
Elapsed1763.641620s; max paired timing perturbation0.097577percent and max single
call fraction0.023810. Both sessions are complete according to raw terminal
records and complete-grid validation. Same Arm NUMA3/CPU319 foreground1T,
full_n_team_stripes (1,0,0,1,1), W13/W2 8MiB/4MiB, Ntile16, BF16 SVE256;
background4T/M48/W13 copies32, foreground copies4, PMUoff,500ms,5warmup31formal.
Binary/source identities and launch command remain as recorded above.

Inline .venv/bin/python imported analyze.analyze, verified both sessions and
formed same-round joint-minus-solo shape contrasts. Durable derived artifact:
tmp/joint_cost_model_20260911/receiver_block_timing/repeatability_decomposition.json.
At8+8, session1 -> session2 prefix/tail/whole contrast medians in us:
- M48->52 W13: -6.560/292.049/283.550 -> -8.020/296.734/287.480.
- M48->52 W2: 5.894/139.815/147.090 -> 2.066/138.416/142.100.
- M96->100 W13: -140.540/244.073/103.970 -> -144.200/251.788/115.220.
- M96->100 W2: -18.483/91.334/74.240 -> -6.460/85.400/78.530.
Session2 M96->100 W13 prefix/tail/whole MAD30.820/2.242/40.280us;
W2 corresponding21.570/5.744/26.000us. Medians need not sum. An initial
arbitrary100ns four-term accounting check failed because native JSON rounding
accumulates. Recheck using sum of four original validated per-call tolerances
(max(2ns,2e-5*mean)) passed; maximum residual142.588ns/133.865ns in sessions1/2.
This changes no native measurements or analyzer acceptance thresholds.

Decision: strong-background W13 prefix cancellation repeats; whole-M differences
must not calibrate isolated tail sensitivity. W2 prefix cancellation is noisier
and its exact magnitude is not stable. At6+6 M96->100 W13 prefix median changes
-11.100 ->0.210us (MAD15.910/18.990us), so no general pressure-independent
negative prefix correction is justified. No model coefficient fit or production
change. Next independent intervention: hold within-call block sequence fixed
while varying B-copy reuse interval, separating inter-call history from added-tail
history. Repeated attribution does not establish that mechanism; dynamic overlap,
unseen full-plan acceptance and equal-budget complete planner search remain open.


### Next intervention: fixed-footprint inter-call gap contract

Read-only CodeGraph query did not locate ignored receiver_block_timing source;
targeted source audit confirmed copies is hard-restricted to4 at input and
weight selection is iteration%copies. Allocations already contain32 copies,
while only4 are touched in each active sequence. Merely admitting more copies
would change the active footprint as well as reuse distance. Remote pgrep of
receiver_block_timing measure.py/foreground_native/run.sh returned no matches
(exec54123 exit1); complete raw artifacts remain authoritative despite the old
SSH observation handle not closing promptly. Do not restart the completed run.

Prepared next intervention contract (not yet implemented/launched):
- Isolated new Lab directory receiver_reuse_gap; retain completed source/raw data.
- Foreground1T CPU319, M96/100, W13 only, four B copies, stride1, identical
  complete-call block sequence and original allocation. Focus on the repeated
  strong-pressure prefix cancellation before expanding to other kernels.
- Requested inter-call gaps0/250/1000us, outside outer GEMM timing. Use a bounded
  local spin with stop checks, no explicit memory-stream background in the gap.
  Gap0 must bypass the delay path. Preserve64 completed-call lead-in and barriers.
- Record actual call end-to-next-start and same-copy previous-end-to-current-start
  intervals on the same selected calls as block costs. Keep per-copy last-end
  state across warmup and exclude undefined first reuse observations explicitly.
  Requested delay is not a substitute for measured reuse interval.
- BG unchanged4T/M48/W13/copies32 at0+0 and8+8. Timing0/1 paired controls remain:
  2M x3gaps x2placements x2timing modes =24 cells per round,500ms,5warmup31formal,
  two independently shuffled sessions. New protocol seeds616011/616012.
- First native numerical/CPU/block-accounting/gap sanity checks, then one-round
  smoke. Formal launch requires complete-grid validation, count resolution<=5%,
  paired timing perturbation<=5%; estimate actual whole run from smoke.
- Analyze joint-minus-solo block increments separately for each gap. Primary
  contrast is M100 first8 blocks minus M96 first8 blocks; report added4-row tail
  separately. Predeclare all three gaps and both sessions, no selected-best gap.
- Also report foreground call rate and both background team throughputs. A gap
  changes foreground average traffic and can alter shared-resource equilibrium;
  constant background configuration does not guarantee constant realized pressure.
  This experiment tests sensitivity to inter-call scheduling at fixed footprint,
  not a clean proof of cache eviction or isolated memory latency. No frequency
  measurement is available, so frequency effects remain an alternative mechanism.
- If prefix cancellation changes reproducibly while background throughput remains
  within repeated variation, proceed to a separate active-footprint intervention.
  If throughput shifts materially, separate load feedback before fitting a history
  coefficient. If cancellation persists, inspect within-call state/response first.

No model equations, coefficients, production defaults or new benchmark execution
changed by this contract. Independent unseen prefix lengths and dynamic/full-plan
validation remain required; these diagnosed shapes cannot become fresh holdouts.


E-class receiver_reuse_gap foreground implemented in isolated optional Lab path.
Extends command with gap_us0/250/1000; restricts M96/100,W13,1T,four copies.
Worker-owned prior-call/per-copy timestamps retained from lead-in; bounded spin
outside begin/end with relaxed stop polling (termination only, no payload).
Gap0 bypasses spin. Post-stop audit recomputes each recorded gap and four-call
reuse interval from raw timestamps, then aggregates only original completion-edge
selected calls. Selected undefined history rejected after64-call lead-in.
Timing0/1 retains same added history bookkeeping; reports17-digit JSON precision.
No kernel/packing changes. Review of exact delta against block-timing source and
build.sh syntax passed. Arm build/numerical and paired perturbation still pending;
rollback is this new directory and its record. Existing model equations unchanged.


receiver_reuse_gap Arm build via bash .../receiver_reuse_gap/build.sh completed
(exec24441 exit0). Inline remote .venv/bin/python drove numactl --membind=3
foreground_native through M96/100 xgap0/250/1000us xtiming0/1,30ms windows,
plus repeated M96/gap0/timing1 on the same persistent process:13 cases passed.
Numerical/CPU flags, commanded identities, all selected history defined, gap
minimum, reuse greater than inter-call interval, finite statistics and block-sum
accounting (1e-12 relative/1e-6ns absolute) passed. Native independently audited
every stored history value against original prior timestamps. Five invalid inputs
(negative/unsupported gap, W2, M48, timing2) rejected; exec30634 exited0.
Fetch exec18886 exited0; local native_correctness.json/build_identity.txt exact
identity and source SHA checked. Build identities:
c3dd64a5e60422997b6464c82403acbf1568725b1592ae677b4a623bd8f03d28  tmp/joint_cost_model_20260911/receiver_reuse_gap/foreground_native
9a5c2f425234c1fbb6c3414b5eb4da903465bb08d07689269062aaf63e8304a9  tmp/joint_cost_model_20260911/receiver_reuse_gap/foreground.cpp
1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629  csrc/moe/arm/sve_bf16/jit_kernels.cpp

These30ms checks do not establish formal count resolution or perturbation.
Next:24-cell paired collector/analyzer, corruption tests and smoke before formal
measurements. No coefficients changed and no performance improvement claimed.


receiver_reuse_gap paired collector/analyzer implemented by bounded copy of the
validated block collector. Grid M96/100 x W13 x0+0/8+8 xgap0/250/1000us xtiming0/1
=24cells. Retains native identities, persistent roles, ARMED/GO/DONE/ACK ordering,
exclusive raw outputs, shuffled rounds,500ms windows and foreground/BG CPU maps.
Analyzer adds exact history-call coverage, no undefined history, finite interval
statistics, gap lower bound and reuse ordering, alongside prior full-grid/native/
block accounting and5percent perturbation/count gates. Protocol native identity
matches tested binary; formal seeds616011/616012,5warmup31formal each.
Command .venv/bin/pytest -q tmp/joint_cost_model_20260911/receiver_reuse_gap/test_protocol.py
passed15 tests0.15s; Ruff on measure/analyze/test passed; run.sh syntax passed.
Sync exec46450 exited0. Arm smoke launched exec24182, seed616001,1/0 rounds/warmup,
raw smoke.jsonl then analyze.py --output smoke.analysis.json. No fit or production
changes. Smoke results and duration-based formal estimate pending.


receiver_reuse_gap smoke exec24182 exited0,24cells/seed616001/1formal0warmup.
Fetch exec94090 exited0. Local original-raw analyze passed, full normalized JSON
remote parity and exact protocol/seed/count identity verified. Max paired timing
perturbation0.292102percent; maximum single-call fraction0.025641; both gates pass.
Elapsed31.394958s gives two5/31-session estimate37.673950minutes (72 rounds).
Formal run.sh launched exec44970 within existing authorization; seeds616011/616012,
24cells/500ms, PMUoff; all source/binary/protocol identities frozen. Await complete
raw and repeatability/throughput feedback assessment before any history fit.


### Reuse-gap session1: gap-zero reference does not reproduce cancellation

First session stdout31/31 and no stderr; fetch exec26555 exited0. Original raw
analyze passed all gates, normalized full remote-analysis parity, exact frozen
protocol and31/5/616011 identity. Elapsed1119.341945s; max paired instrumentation
0.035402percent; maximum single-call fraction0.025. Inline local .venv/bin/python
formed same-round M100-minus-M96 joint-minus-solo decomposition at8+8 and wrote
receiver_reuse_gap/session1.decomposition.json. Per-round prefix+tail+overhead
matches whole contrast within1e-6ns. Derived per-shape summary keys use *_us for values divided by1000; raw native
keys remain nanoseconds. No fitted parameters.

Requested gap0/250/1000us: prefix contrast medians -6.308/-8.075/-10.030us
(MAD9.670/10.242/7.051), added-tail241.811/239.001/238.985us
(MAD3.001/4.176/2.557), whole233.288/230.331/229.297us.
Observed M96 joint same-copy reuse32.674/33.707/36.709ms, M100
34.705/35.724/38.721ms. Gap minima/history checks passed. Per-background-role
aggregate call rates remain roughly5496-5507calls/s across shapes/gaps; no large
throughput change apparent, but no physical DRAM-rate measurement in this probe.

Critical reference failure: gap0 does NOT reproduce prior W13 prefix cancellation
-140.540/-144.200us. Thus current gap contrasts cannot establish why the earlier
cancellation occurred. New-binary/session comparisons of paired joint-minus-solo
incrementus for M96: old timing0 sessions1/2=1139.980/1209.310, old timing1=
1182.550/1215.930; new gap0 timing0/1=1042.907/1040.753. M100 old timing0=
1291.300/1301.100, old timing1=1293.900/1307.700; new gap0 timing0/1=
1279.733/1277.631. Solo times remain approximately9844-9848us(M96) and
10282-10289us(M100). Main change is M96 competition response, present in both
new timing modes. Paired instrumentation acceptance only bounds boundary-clock
overhead; it cannot rule out shared bookkeeping/binary-layout/session effects.

Decision: retain all gaps and finish session2 unchanged. Before interpreting reuse
history, compare the two frozen old/new binaries at gap0 under randomized paired
same-session conditions. This separates binary/probe effect from between-session
state drift; it is still not direct hardware-mechanism attribution. No history
coefficient or negative prefix correction accepted. Full-plan generalization and
equal-budget planner search remain outstanding.


### Reuse-gap session2: repeatability limits history fitting

Transfer exec37085 exited0. Original session2 raw has terminal complete and all
expected cells. Local analyze passes all gates, full normalized remote-analysis
parity, exact frozen protocol and31/5/616012 identity. Elapsed1119.057652s;
max paired timing perturbation0.142149percent, max single-call fraction0.025.
Derived session2.decomposition.json uses microsecond labels, same-round contrasts;
per-round accounting residual below1e-6ns. Both sessions complete; no fit.

M96->100 W13 at8+8, requested gaps0/250/1000us, session2 medians:
prefix -1.025/-17.030/-83.541us (MAD8.300/11.991/49.046),
added-tail248.870/245.486/246.020us (MAD4.057/2.889/2.387),
whole247.083/232.628/156.403us (MAD10.240/10.926/44.377).
Session1 corresponding prefix -6.308/-8.075/-10.030us; hence strong gap1000
prefix cancellation is not stable across sessions. Gap0 in both sessions fails
to reproduce prior old-binary prefix contrast about-141/-144us.

M96 same-round joint-minus-solo increment median/MADus, timing0 then timing1:
session1 gap0 1042.907/6.601,1040.753/7.846;
gap250 1055.519/7.405,1053.700/4.953;
gap1000 1054.862/6.152,1054.964/6.900.
session2 gap0 1053.401/7.501,1052.204/8.635;
gap250 1065.214/11.491,1070.523/5.637;
gap1000 1106.280/45.899,1134.655/55.264.
Thus increased gap1000 spread is present in both timing modes. Session2 M100
joint means remain11579.5/11589.2/11582.2us, MAD8.12/7.10/8.40us; M96
gap1000 joint mean10980.4us,MAD53.29us. All solo means remain stable within
a few microseconds. Background role aggregate rates about5476-5491calls/s in
session2 versus5496-5507 in session1; these are call rates, not direct DRAM rates.

Decision: no deterministic reuse-history coefficient accepted. Complete the
same-session frozen-old/new gap0 paired control before assigning the reference
shift to instrumentation, binary layout or session state. Keep gap1000 variability
as a measured uncertainty requiring explanation, not a fitted negative correction.
Whole-model unseen-plan and equal-budget search requirements remain unchanged.


### Frozen-probe paired control prepared

E-class isolated collector receiver_probe_pair uses unchanged block_timing and
reuse_gap foreground binaries (SHA6adf8fee... and c3dd64a5...), unchanged background.
No rebuild. Four persistent processes; only one foreground is commanded per cell,
the other has no active workers after ACK. Separate allocations remain a possible
probe/process-layout confound. Retains64-call lead-in, ARMED/GO/DONE/ACK and native
correctness, original geometry/NUMA3/foreground319,500ms PMUoff observation.
Grid M96/100 xW13 x0+0/8+8 xold/new xtiming0/1=16cells; new receives gap0.
Both sessions planned5/31 seeds616111/616112. Primary outcomes: within-round
probe difference in joint-minus-solo response, prefix/tail and whole-stage, plus
background throughput and timing controls. No selected-best version or refit.
Repeat sessions restart foreground processes, but cannot uniquely separate binary
code from address-layout effects. Opposing foreground activity is excluded by the
collector's per-cell role map. No production or model equation changes.

Reused impact/test/review scope: only new collector/analyzer/tests/protocol/run
under ignored Lab root. CodeGraph previously failed to locate these ignored
collectors; bounded source copies and focused reads used. Command .venv/bin/pytest
-q tmp/joint_cost_model_20260911/receiver_probe_pair/test_protocol.py passed10
checks0.18s (full grid, corrupt blocks/cells/CPU/NaN/mode, perturbation, wrong
foreground role and missing new-probe history); Ruff and run.sh syntax passed.
Next16-cell smoke before formal launch; native correctness evidence reused only
because both binaries are unchanged and runtime hashes are checked on startup.


Frozen-probe smoke sync exec51963 exit0, Arm smoke exec90865 exit0,
fetch exec58154 exit0. Local original raw analyze passes full normalized remote
parity, exact protocol and1/0/616101 identity;16cells complete. Max paired timing
perturbation1.491015percent, max single-call fraction0.023256, both gates pass.
Elapsed20.616370s yields two5/31-session estimate24.739644minutes. No response
conclusion or fit from smoke. Formal run.sh launched exec72533 within existing
experimental authorization: seeds616111/616112,16cells/500ms,5warmup31formal,
unchanged foreground binaries. Await both complete sessions and paired response
attribution; source/protocol frozen during measurement.


### Frozen-probe pair session1: no stable140us version offset

Fetch exec41403 exited0. Complete raw session1 reanalysis passed all gates,
full normalized remote-analysis parity, exact protocol and31/5/616111 identity.
Elapsed731.898601s; max paired timing perturbation0.075251percent; max single-call
fraction0.023256. Inline local .venv/bin/python wrote session1.comparison.json;
all increments are within-round joint-minus-solo, version differences are within
round new-minus-old increments, unitsus. Separate medians need not subtract.

M96 old/new increment median(MAD): timing0 1061.760(22.680)/1080.697(18.003),
timing1 1124.470(58.510)/1076.153(14.834).
Paired new-minus-old: timing0 +9.859(MAD43.502), timing1 -15.239(MAD52.989).
M100 old/new timing0 1297.600(6.500)/1300.810(7.624), timing1
1298.800(7.100)/1301.962(7.439); paired version differences+1.178(10.143)
and+4.331(11.801).
M96->100 timing1 prefix/tail/whole medians: old -79.780/252.828/175.870us,
new -15.249/247.239/229.746us. Prefix MAD52.210old/15.265new;
tail MAD2.763old/2.194new. Thus M96 response remains variable even with frozen
old binary; first paired session does not support a constant140us version offset.
Within-version whole-stage perturbation gate does not resolve competition-increment
uncertainty. Await session2; no binary-code causality or history coefficient claimed.
Separate foreground allocation/layout and scheduling phase remain possible factors.


### Frozen-probe pair session2: constant version correction rejected

Fetch exec59545 exited0. Complete raw session2 reanalysis passed all gates,
full normalized remote parity, exact protocol and31/5/616112 identity. Elapsed
731.975651s; max timing perturbation0.157157percent; max call fraction0.023256.
Same inline .venv/bin/python paired analysis wrote session2.comparison.json.
M96 old/new increment median(MAD)us: timing0 1054.660(22.230)/1044.999(26.501),
timing1 1065.730(34.840)/1044.483(17.609). Paired new-minus-old timing0
-18.410(MAD56.313), timing1 -23.312(MAD60.061).
M100 old/new timing0 1275.100(5.900)/1262.808(6.579), timing1
1285.500(6.600)/1271.886(6.974); paired differences -7.864(9.548),
-10.518(9.690). Medians of differences are not differences of medians.
M96->100 timing1 prefix/tail/whole: old -24.040/241.630/219.020us,
new -15.957/243.726/224.102us. Prefix MAD37.800old/18.783new;
tail MAD2.906old/2.690new.

Both paired sessions fail to show a stable140us binary-version offset. Old-probe
prefix contrast changed -79.780 ->-24.040us, versus original separate sessions
about-141/-144us. This disproves using that earlier contrast as a fixed prefix
correction, but does not establish zero instrumentation impact or identify a
hardware mechanism. M96 competition variability remains far larger than solo
variability; M100 and added-tail increments are comparatively stable.
No version-specific or reuse-gap coefficient added. Next read-only diagnostic:
check paired response variation against measured background call rates and
round/order; if existing observables do not explain it, use a controlled start-phase
intervention before fitting a timing-history law. This is needed to distinguish
predictable dynamic overlap from residual uncertainty. Real-plan residuals remain
excluded from calibration, and unseen-plan/equal-budget-search gates stay open.


### Existing-observable diagnostic and phase-control audit

Inline local .venv/bin/python revalidated both receiver_probe_pair raw sessions
and full remote-analysis parity, then wrote variation_diagnostic.json and
variation_transfer_diagnostic.json. No production/model parameters changed.
For each M/version/timing group, response is within-round joint-minus-solo us;
features are joint-window aggregate local/cross background calls/s, round index,
and cell position within round. Pearson correlations are descriptive,31points per
group, not causal or a comprehensive test of nonlinear dependence. M96 correlation
with local/cross throughput ranges -0.166..0.405 across sessions/groups; signs and
magnitudes are inconsistent. Round/position do not give a uniform trend.

A diagnostic ordinary least-squares fit (intercept plus all four features,
standardized on session1 only; no feature selection or tuning) trained session1
and transferred unchanged to session2. M96 second-session MAEus for regression
versus session1-median constant: old timing0 61.806 vs49.945, old timing1
64.535 vs62.731, new timing0 50.707 vs45.123, new timing1 51.869 vs38.577.
All four are worse than constant. This is retrospective diagnostic transfer,
not untouched model validation; failure rejects this simple explanation, not
all possible dependence on observed features. No coefficients adopted.

Focused source audit of background.cpp and block_timing/foreground.cpp confirms
launch precedes64 completed-call lead-in and ARMED; workers continue while main
waits for GO. GO only defines observation window. Changing GO delivery delay alone
would shift observation selection, not impose a known kernel start phase.
Therefore do not use controller GO-delay as a causal phase intervention.

Next minimally invasive evidence path: both native probes already retain complete
call begin/end records; foreground additionally retains block boundaries. Export
these existing records after stop, with absolute window endpoints and original
team/copy indices, without adding timed-loop clocks or barriers. Preserve a matched
export-off control and exact source audit of timed loops. Join actual foreground
block intervals with background team call intervals to observe phase/coherence;
background call boundaries are only phase proxies, not measured per-block memory
requests. This observational step can identify what a subsequent controlled launch
intervention must change while avoiding a false phase-control claim. Unknown
clock alignment must be checked on the same host; retain completion-edge selection
and include boundary-crossing records for overlap accounting. Full-plan generality
and equal-budget complete planner search remain the final acceptance requirements.


E-class receiver_phase_trace standalone foreground/background source prepared.
Bounded copies of receiver_block_timing/foreground.cpp and
receiver_locality_response/background.cpp; appended trace0/1 command field.
After worker cleanup and existing numerical/statistical checks, optional JSON
records export [team,iteration,copy,begin_ns,end_ns,boundaries] for all retained
calls, including lead-in and boundary-crossing calls. Background boundaries empty;
foreground boundaries only when original timing1 is enabled. Absolute observation
window endpoints emitted for both trace modes. Existing record allocation/capacity,
clock reads, worker launch/lead-in/GO/ACK/cleanup remain byte-identical: local
inline .venv/bin/python asserted complete source region equality from launch
atomic declarations through just before post-stop duration aggregation, both files.
This source equality does not prove identical generated layout; matched trace-off
controls and target checks still required. No production APIs or formulas changed.
Build.sh syntax passed; target build and raw-record reconstruction pending.
Rollback boundary is the new optional Lab directory and its report. Reused shared
impact-analysis/test-selector/code-review-gate and C++ scope; no kernel refactor.


Phase-trace Arm build exec27961 exited0 for foreground/background; validator sync
exec85453 exited0. Inline remote .venv/bin/python drove16 short30ms configurations:
foreground M96/100 x timing0/1 x trace0/1; background4T/M48/W13 xteams1/8
xcpu_end316/276 xtrace0/1. Each persistent role ran all its cells. Native numerical
and CPU checks plus validate_trace reconstructed completion counts, median,
foreground mean/max/block means, background per-team medians, consecutive
iteration/copy identities and non-overlap/boundary enclosure; all16 passed,
exec43035 exit0. Fetch exec67728 exit0; local revalidation all16 plus commanded
identities and source/build hashes passed. Build identities:
6f964c1c30c138f20c94f69e77fa1156a8d2bebabf53c2350911d55d6ba95177  tmp/joint_cost_model_20260911/receiver_phase_trace/foreground_native
de11b3770e4a3f037c78de0873645c92720d2df012dadba27cc83cdc0fb32b9c  tmp/joint_cost_model_20260911/receiver_phase_trace/foreground.cpp
f37b02397501ca38e6d3910a94332f0df1c15fc5bc28a7ed552d9b5d70bd200f  tmp/joint_cost_model_20260911/receiver_phase_trace/background_native
e3ab70990907c7b20fc3955d21fe89896a5d9a4a3c409a48b9096bf35853dce9  tmp/joint_cost_model_20260911/receiver_phase_trace/background.cpp
1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629  csrc/moe/arm/sve_bf16/jit_kernels.cpp

No concurrent foreground/background clock-alignment validation yet; paired export
perturbation and formal phase observations pending. Short correctness samples
are not calibration/performance data. Validator Ruff passed; no model coefficients
changed. Next add corruption coverage and matched concurrent collector preserving
native completion-edge accounting, then smoke quality gates.


Phase-trace collector/analyzer complete: M96/100 xW13 x0+0/8+8 xtrace0/1=8cells,
foreground block_timing1 in both modes. Native summary reconstruction applies to
trace1; per-background-team first/last retained calls must span all selected
foreground calls. This checks same-host time-coordinate consistency/coverage,
not independent clock calibration. 9 protocol/corruption tests passed0.24s,
including valid native BG counts but insufficient cross-role coverage; Ruff passed.
Sync exec52073 exit0; Arm smoke exec99069 exit0, fetch exec84750 exit0.
Local original-raw reanalysis matches normalized remote analysis completely;
exact protocol and1/0/616201 identity checked. Max paired export perturbation
0.711042percent, max single-call fraction0.023810; all gates passed.
Smoke10.456118s estimates two5/31 sessions12.547341minutes and58.640762MiB raw.
Formal run.sh launched exec14584 within existing authorization, seeds616211/616212,
500ms windows, frozen protocol/binaries. No phase-performance conclusion from
smoke; wait complete raw and validate actual interval features before fitting.


### Phase-trace session1: complete time coordinates and within-window variation

Fetch exec47485 exited0. Local original-raw analyze passed all gates and full
normalized remote-analysis parity, exact protocol and31/5/616211 identity.
Elapsed366.033575s; max paired export perturbation0.012971percent; max single-call
fraction0.023256. Trace1 summary reconstruction and background per-team coverage
of all selected foreground calls passed. Same Arm NUMA3 CPU319 1T foreground,
W13 full8MiB stripe, Ntile16 (1,0,0,1,1), BG4T/M48/W13 copies32 at8+8.
Same-round joint-minus-solo increment median/MADus: M96 trace0 1046.930/6.330,
trace1 1044.720/5.100; M100 trace0 1279.400/7.000,trace1 1278.400/5.700.
Recorded session1.summary.json. No new history/phase fit.

Raw selected-call follow-up in session1.copy_diagnostic.json: median within-window
call MADus solo/joint M96 3.305/30.690, M100 4.335/35.730; median within-window
rangeus M96 25.300/192.990, M10035.710/244.030. Thus stable window averages do
not imply constant per-call response. Joint copy0..3 mediansus (median across
rounds of within-copy medians): M96 10887.810/10875.960/10895.060/10900.530;
M10011575.010/11555.340/11558.725/11583.350. Solo copy medians M96
9846.110/9846.140/9844.795/9845.295; M10010285.905/10288.075/10293.475/10286.675.
These descriptive copy differences do not establish an address-layout mechanism.
Next join foreground block starts to actual background call histories. Use only
past call lengths for any predictive phase feature; dividing by the current BG
call's future measured duration would introduce outcome leakage. Keep retrospective
phase description separate from prospective model features. Await session2 before
making repeatability or transfer claims; full-plan/search acceptance stays open.


Historical phase-feature diagnostic implemented locally in phase_features.py.
At each observed foreground block start, each BG team uses latest known call start
and median duration of its last8 completed calls; no current/future completion
length enters phase. Per LLC features: active fraction, mean cos/sin of age divided
by historical duration, mean historical duration. These are call-phase proxies,
not memory demand and not autonomously predicted planner state. Four tests passed
0.12s: future-end invariance, completed-history sensitivity, gap/start boundary,
insufficient history; Ruff passed. Existing timed/native/collector code unchanged.

Local original session1 raw analysis/full parity rechecked; extraction exec52641
exit0 produced23357 selected block observations in session1.phase_features.jsonl,
including copy/block identity and actual block duration. First-session OLS per
(M,block),17groups, intercept+all8 features, standardization from first-session
only; zero-variance dimensions use scale1. No feature selection/tuning. Frozen
session1.frozen_phase_diagnostic.json written before reading session2 data.
Training block MAE14.260840us versus per-group constant-mean14.958281us; training
improvement is not validation. Second session latest read14/31, stderr empty.
Next extract same features and test frozen coefficients against same constants on
session2, grouped by M/full/tail. This checks repeat transfer given observed starts,
not unseen-shape generalization or complete-plan prediction. No production/candidate
cost-model coefficients changed; full-plan and equal-budget search gates remain.


### Phase-trace session2: frozen conditional diagnostic transfer

Second session completed (31 formal/5 warmup, seed616212,500ms, same frozen
protocol and binaries). Local inline .venv/bin/python imported analyze.analyze
and phase_features.background_features, revalidated original session2.jsonl,
checked full normalized session2.analysis.json parity and exact protocol/seed,
and recomputed every selected block prediction from the unchanged
session1.frozen_phase_diagnostic.json. All previous session2.phase_validation.json
metrics reproduced at rtol1e-10/atol1e-9. Audit and source/model SHA256 saved in
receiver_phase_trace/session2.phase_audit.json; local exec73371 exited0.
Elapsed365.846267s, max paired export perturbation0.031123percent,
max single-call fraction0.023256; all measurement gates passed.

Arm NUMA3: 1T foreground CPU319, M96/100 W13, four weight copies, full8MiB
stripe, Ntile16; 8 local plus8 cross-LLC background4T/M48/W13 teams,32copies.
Both trace modes retain foreground block timing. Frozen first-session OLS
(intercept plus eight historical phase features, separate M/block groups) is
compared against frozen first-session per-group mean duration, not the planner.

| Session2 target | Observations | Constant MAE/us | Phase MAE/us |
|---|---:|---:|---:|
| M96 full blocks |11288|14.0374|14.0102|
| M100 full blocks |10672|14.2197|14.2175|
| M100 four-row tail |1334|26.5356|18.4077|
| M96 sum of block times per call |1411|37.2214|37.4476|
| M100 sum of block times per call |1334|46.5475|39.8917|

Round-level diagnostic averages each round's paired absolute-error difference
(phase minus constant), then uses10000 resamples of31 rounds,seed616213.
M100 tail improves31/31 rounds: mean delta-8.1257us, descriptive95percent
bootstrap interval[-9.4527,-6.7667]. M100 block sum improves30/31 rounds:
-6.6557us,[-7.8939,-5.4469]. Full-block deltas M96-0.0273us and
M100-0.0022us have intervals spanning zero; M96 block-sum delta+0.2241us
also spans zero. These intervals describe this session and assume exchangeable
rounds; they are not independent-session or unseen-shape confidence guarantees.

The tail association transfers to one repeat session, but coarse call phase
provides essentially no full-block improvement. Do not adopt these coefficients:
features require observed foreground starts and past measured background lengths,
M/block-specific fits do not share across unseen shapes, and call sums exclude
outer overhead. This is not autonomous complete-plan prediction, nor evidence
that physical memory request phase has been identified. Retain phase as a tail
sensitivity diagnostic; do not add a universal phase correction or refit failed
full-plan residuals. Next model work must test receiver sensitivity with independent
resource-demand/state features and validate autonomous timeline feedback. Existing
mixed-width failure and unseen-route/equal-budget complete-search gates remain open.

Static review: modified manifest entry parses with yaml.safe_load after quoting
its existing colon-bearing next_decision text. Whole-manifest parsing still fails
at unrelated line785 (earlier unquoted prose); that entry was not changed. Remote
pgrep for receiver_phase_trace foreground/background/measure/run found no matching
process (SSH exec80762 exit1), consistent with both complete raw sessions.
No new benchmark, production code, model equation or default changed in this audit.


### Active DRAM response path and input granularity audit

Read-only source/fixture audit after phase diagnostic closure. CodeGraph did not
uniquely index the ignored dynamic_resource_candidate/simulate.py; targeted source
inspection followed. The active solver already multiplies request rates by solved
speeds, excludes the receiver's whole expert from peer pressure, iterates a damped
fixed point, checks capacity, and conserves DRAM budgets. Do not diagnose existing
mixed-plan errors as absence of demand feedback.

However queue_enabled replaces the legacy target with
1/(1+dram_queue[stage]*peer_actual_dram_rate). Segment sensitivity,
local_sensitivity and refill_sensitivity are not applied on this path. This is
intentional separation of legacy and physical inputs, not proof those old fields
should be reused as physical sensitivities. A local synthetic two-job diagnostic
changed all three legacy sensitivities to1e6 and obtained identical complete
prediction objects; completion18.1803398878us with w13 queue coefficient0.1.
Saved queue_sensitivity_path_audit.json. This is structural evidence, not hardware
measurement. Focused command:

```sh
.venv/bin/pytest -q tmp/joint_cost_model_20260911/dynamic_resource_candidate/test_queue_receivers.py tmp/joint_cost_model_20260911/dynamic_resource_candidate/test_dram_queue.py tmp/joint_cost_model_20260911/dynamic_resource_candidate/test_dram_clocks.py
```

23 tests passed0.13s. No solver code or coefficients changed.

Frozen receiver_mixed_holdout/jobs.json structural audit saved with source SHA256
in dynamic_resource_candidate/receiver_input_structure_audit.json. All segments
have dram_service_us/service_us=1, an explicit unvalidated window assumption.
Request boundaries are sparse cumulative measurement/interpolation nodes, not
necessarily actual kernel block boundaries. Some segments span many complete
blocks and a tail. Node label measured describes the cumulative node provenance,
not independent validation of the temporal allocation between nodes.

| Plan | Request nodes/segments | Measured cumulative nodes | Segments spanning >12 rows |
|---|---:|---:|---:|
| anchor |908|456|330|
| mixed_narrow |988|536|402|
| mixed_wide |928|476|342|

Every plan has83 experts with M<12 using unvalidated M12 request anchors. Their
fraction of summed isolated GEMM service ranges3.76-6.69percent across width/plan
groups; this is work summed over parallel teams, not an upper bound on critical-path
or interference error. Do not attribute the mixed-plan underprediction to these
points from counts alone.

Next bounded model experiment: use the existing controlled block-timing grid to
compare shared full-block versus independently calibrated tail response, holding
T0 and independently measured background pressure fixed. Preserve request budgets
and distinguish a new physical-response sensitivity from legacy fields. Test
cross-M/history transfer and pressure transfer before adding the parameter to the
active solver. A future block split must keep cumulative bytes/time conserved and
state explicitly where within-segment allocation is assumed. Existing phase OLS
will not be used as a substitute for this calibration. Mixed-width full-plan and
equal-budget complete-search acceptance remain required.


### Retrospective block-response pressure/history transfer

Local inline .venv/bin/python revalidated receiver_block_timing/session1/2 raw
through analyze.analyze, exact frozen protocol,31/5/615911-615912 identity and
full normalized remote-analysis parity; all gates passed. Independently revalidated
receiver_pressure_curve session1 using fit_response.load_session: BG-only read
rates6+6=224.811440695 and8+8=275.785924877 GB/s. Pressure reference remains295
GB/s, not newly measured capacity. Raw hashes recorded in
receiver_block_timing/block_response_transfer_diagnostic.json.

Retrospective experiment on previously viewed data; not untouched validation.
Fit only block-timing session1 M48/52 at6+6, with W13/W2 separate. Freeze each
shape/block T0 from session1 solo. Response delta is the median of31 same-round
joint-minus-solo block mean durations. Model delta=T0*a_group*rho/(1-rho),
nonnegative coefficient, zero intercept, equally weighted relative block-node
responses. Shared model groups first/reuse (tail belongs to reuse); split model
groups first/reuse/tail4. No target history or8+8 response used in fitting.
These are conditional response diagnostics, not changes to the active solver.
BG rates come from a different probe session/protocol; transfer of that pressure
proxy is an explicit unresolved assumption.

Fit coefficients first/reuse/tail4: W13 .00769337/.00760717/.05573582;
W2 .00508904/.00609548/.05571397. Without tail separation, reuse coefficients
are .01448269/.01318384, roughly twice the full-block-only estimates. Each tail
coefficient has only one training shape/history/pressure node;31 rounds are
repeat observations, not31 distinct calibration environments.

Session2 response MAEus; full-block scores average distinct block nodes, tail
scores contain one tail node per stage/test, not a population error estimate:

| Transfer | Stage | Block | Shared reuse | Separate tail |
|---|---|---|---:|---:|
| M48/52,6+6 to8+8 |W13|full|79.32|16.27|
| M48/52,6+6 to8+8 |W13|tail4|205.66|53.74|
| M48/52,6+6 to8+8 |W2|full|31.03|25.73|
| M48/52,6+6 to8+8 |W2|tail4|98.17|31.68|
| M48/52 to96/100,same6+6 |W13|full|26.35|3.21|
| M48/52 to96/100,same6+6 |W13|tail4|43.61|14.21|
| M48/52 to96/100,same6+6 |W2|full|12.32|1.39|
| M48/52 to96/100,same6+6 |W2|tail4|26.77|2.10|
| M48/52 to96/100,6+6 to8+8 |W13|full|99.53|12.30|
| M48/52 to96/100,6+6 to8+8 |W13|tail4|160.82|98.28|
| M48/52 to96/100,6+6 to8+8 |W2|full|43.08|17.79|
| M48/52 to96/100,6+6 to8+8 |W2|tail4|45.29|84.10|

Separate tails reduce contamination of the shared full-block response, but one
constant relative tail multiplier does not transfer reliably across both history
and pressure. At8+8 M100 W2 tail predicted169.504us increment versus85.400us
observed; W13 predicted350.067us versus251.788us. No adoption on this evidence.

Descriptive effective tail coefficients saved in tail_effective_response_diagnostic.json.
Session2 M52 W2 coefficient changes .056699 at6+6 to .045338 at8+8;
M100 W2 changes .052614 to .028070. M52/M100 W2 solo tails are212.704/211.964us
(from frozen session1). Thus similar isolated tails do not imply identical
conditional response in this protocol. These effective coefficients include any
pressure-proxy and protocol mismatch and are not identified physical constants.
Prior M96-prefix variability also prevents treating all history differences as
stable hardware effects. No total-time or planner improvement claimed.

Next discriminating experiment must cover an intermediate pressure with matched
BG-only measurement and a tail history excluded from fitting, then validate across
sessions before integration. Keep T0 fixed; do not fit a multi-parameter history
and pressure surface from the single tail training node or from failed full-plan
residuals. Scope remains full autonomous mixed-plan prediction and equal-budget
complete planner search, both still outstanding.


### Matched tail-response experiment preparation

E-class standalone probe receiver_tail_matched prepared using shared
impact-analysis/test-selector/code-review-gate; current ignored Lab files are not
indexed, so inspected bounded source directly. New foreground.cpp differs from
receiver_block_timing only by accepting M76. Reverse substitution reproduces the
entire source exactly; no timed loop, allocation, record layout, launch, GO or
kernel changed. Existing11-boundary record supports M76's seven blocks. Build
script changes only its isolated output directory; bash -n passed. No production
API, numerical contract, build or model equation change. Rollback is this optional
Lab directory. Design.json records source hashes and predeclared split:
training M52/100 at0/6+6/8+8, validation M76 at0/6+6/7+7/8+8 and7+7 for training
shapes; W13/W2 separate. Fit session1 only; session2 repetition. Training and
coefficient freeze must precede validation acquisition. BG-only including idle,
PMU off/on and block timing off/on controls required in the matched protocol.
Collector/analyzer and smoke runtime estimate remain pending; no formal run yet.
Arm source sync exec9321 exited0; isolated build exec61267 launched.

Arm build exec61267 exited0. Native short correctness exec19136 exited0:
M52/76/100 xW13/W2 xtiming0/1 all12 numerical/CPU/block-accounting checks passed;
unsupported M75,invalid stage2 and timing2 rejected. Fetch exec92884 exited0.
Local revalidation checked exact12-cell coverage,all negative results,raw source
and binary build identities,and block accounting. Foreground binary SHA256 afdf7d5ea495eee3ab5445d0a8f8bab054166658f92963e799524299d54ce60c; source 95b266ed0112928fd8d4c7cc0c3bd3ef7cef8b5ccbe8fdc3bd739c06e861cb8b.
Modified manifest entry YAML and scoped diff checks passed. No calibration or
performance result from these30ms samples. Next implement matched collector and
raw analyzer with training/validation separation, then smoke before formal data.


Matched collector/analyzer implemented in receiver_tail_matched, bounded copies
of receiver_pressure_curve PMU orchestration and event definitions. Six-field
cell=(M,stage,local,cross,block_timing,PMU) independently pairs both instruments;
BG-only/idle has timing0 only. Train54 and validation72 cells per round; train
excludes M76 and7+7. Validation collector requires a frozen-model file and saves
its SHA256; file presence is a provenance guard, not proof of fit validity.
Training/validation sequencing remains required. Collector/native READY-ARMED-GO-
DONE-ACK flow and counter reset semantics preserved. NUMA3 binding explicit.
Analyzer checks both grids, roles/CPU identity, numerical flags, window lengths,
block-count/accounting, finite counters, idle correction and both instrument
perturbations. Existing limits5percent instrumentation and5percent single-call
fraction retained. Six focused tests passed0.21s, covering both complete synthetic
grids, missing/duplicate samples, corrupt blocks/accounting, missing frozen model,
CPU/event coverage and exact native whitelist-only change. Ruff passed after format.
No response coefficients or production/model equations changed.
Remote matching-probe pgrep exec66513 exit1 found none; sync exec95676 exited0.
Training-only smoke exec39900 launched: numactl --membind=3 .venv/bin/python
receiver_tail_matched/measure.py --split train --seed616301 --rounds1 --warmup0
with exclusive smoke.jsonl output; analyzer follows on success. Validation shapes
and pressure are not in smoke. Wait actual gates and runtime before formal runs.


Matched train smoke exec39900 exited0,fetch25456 exited0. Local original-raw
analyze reproduced full normalized remote analysis and exact protocol/1/0/616301
train identity. All gates passed: max paired PMU perturbation1.113724percent,
block timing0.877836percent,single-call fraction0.023256; elapsed53.779764s.
Smoke audit estimates two training5/31 sessions64.535716minutes and all training
plus validation sessions150.583338minutes; validation is conditional on frozen
training parameters and is not launched. Estimates are not performance results.
run_train.sh passes shell syntax, uses exclusive native/raw outputs and shell
noclobber for logs. Synced exec97339 exit0; formal training launched exec60751
under existing Arm experiment authorization, seeds616311/616312,54cells each
round,5warmup/31formal,NUMA3. No measured response or model improvement claim.


Training observation extraction implemented locally in reduce_training.py, separate
from the running collector. It accepts only complete train1 (31/5/616311), exact
local frozen protocol and passing raw-analysis gates; rejects validation, smoke,
repeat-session and empty input. It extracts56 M/stage/block/pressure nodes from
PMU-off/timing-on same-round joint-minus-solo samples, preserving all31 paired
deltas,solo times,median andMAD. Matching BG-only PMU-on read/write rates remain
signed independent pressure features. No response parameters fitted and no remote
running files changed. Tests explicitly distinguish median(pair differences) from
difference(medians), exclude PMU-on timing contamination and verify h4/h8 tails.

fit_plan.json predeclares a conditional history/pressure interpolation experiment
before complete training read: absolute block delta, T0 unchanged; stage/history
full-block nodes pooled across available training shapes, tail4 interpolated h4
throughh8, pressure piecewise-linear through zero and measured6+6/8+8 BG-only
read-rate nodes, no extrapolation. Controls are zero response and history-pooled
first/reuse/tail groups with identical pressure interpolation. Signed observations
retained; non-increasing pressure knots rejected. Validation BG-only pressure may
condition a diagnostic prediction but cannot be called autonomous planner input.
This declaration is not an implemented/adopted cost-model equation; fit code and
coefficient freeze remain pending. No validation-family selection allowed.

Live status checked through SSH exec47309: run_train PID2471002, train1 collector
PID2471067 and foreground PID2471068 present; stdout showed first warmup round.
Unified training handle60751 remained live. No restart or duplicate run.

Training reducer focused tests:6 passed0.09s; Ruff passed. Formal raw reduction
not run until train1 completes; no model result inferred from current partial data.


Matched conditional fit_response.py implemented locally while formal training
handle60751 remains live. Primary and zero/pooled-history controls follow the
predeclared fit_plan.json; no model coefficients generated from partial data.
Fit validates exact56-node grid,31 finite paired samples per node, block identity,
summary consistency and increasing independent pressure knots. Main reads raw
train1 through the validated reducer, so validation/repeat/smoke cannot fit.
Prediction rejects unsupported row/history/pressure domains; signed observations
are preserved. Output is exclusively created and records raw/fit-plan/implementation
hashes.13 synthetic tests passed0.13s: training-node recovery, interior bilinear
response, zero pressure, domain rejection, duplicate/missing nodes, inconsistent
summaries, pressure ordering and no clipping. Updated mathematical model's Lab
section with formulas and limitations; no active solver/default change.


Frozen conditional evaluation implemented locally in evaluate_response.py.
Accepts only complete train2 repeat or validation seeds616321/616322, exact
protocol/31/5 and passing native/PMU/timing gates. Validation must carry SHA256
of the supplied frozen model; CLI checks fitting implementation and predeclared
fit-plan hashes. Uses independent same-session BG-only median read pressure,
PMU-off/timing-on same-round block deltas, and all three predeclared families.
Summaries retain total and supported node counts perM/stage/pressure/full-tail;
out-of-domain pressure produces an explicit unsupported record, not extrapolation
or silent exclusion. A group with zero supported nodes has null errors, not zero.
No full-plan or unconditional prediction claim from this evaluator.

Five focused tests passed0.13s and Ruff passed: known-response error, coverage
denominator, null empty support, reject training-fit data/smoke and wrong model
identity. A separate inline synthetic integration check duplicated the validated
54-cell smoke grid into a clearly synthetic31/5 train2 metadata fixture and used
synthetic coefficients. All56 response nodes and three families traversed the
complete evaluator successfully. Saved evaluation_pipeline_synthetic_check.json;
this establishes execution coverage only, not repeated measurements or model
accuracy. Running remote collector unchanged; formal handle60751 still live.


### Matched training session1 complete and parameters frozen

First training session completed31/5/616311; remote process check exec89761
confirmed second-session collector PID2555477 running train2. Fetch exec82047
exited0. Local analyze from original train1.jsonl reproduced full normalized
remote analysis, exact protocol and formal identity; all gates passed.
Elapsed1925.366209s, max PMU perturbation0.135352percent, block-timing
perturbation0.204109percent, max single-call fraction0.023256.

Local commands completed successfully:

```sh
.venv/bin/python tmp/joint_cost_model_20260911/receiver_tail_matched/reduce_training.py tmp/joint_cost_model_20260911/receiver_tail_matched/train1.jsonl --output tmp/joint_cost_model_20260911/receiver_tail_matched/train1.observations.json
.venv/bin/python tmp/joint_cost_model_20260911/receiver_tail_matched/fit_response.py tmp/joint_cost_model_20260911/receiver_tail_matched/train1.jsonl --output tmp/joint_cost_model_20260911/receiver_tail_matched/frozen_model.json
```

Frozen model SHA25615971f62aa393ae3bdf11f6aba7a744902e87533a08361be5763d87eb8f2d13b.
Independent BG-only read knots224.576205333 and276.361017803GB/s.
Fit recomputation from saved observations matched every generated model field;
training raw SHA checked. No negative fitted full/tail nodes. freeze_audit.json
records identity and descriptive training tails (competition delta us):

| Stage | History |6+6|8+8|
|---|---:|---:|---:|
|W13|4|83.180|293.519|
|W13|8|63.948|248.201|
|W2|4|37.885|136.525|
|W2|8|37.740|91.227|

W2's training tail-history difference is small at6+6 and larger at8+8; this is
training evidence only. M76/7+7 validation remains uncollected, and train2 is
still running. Freeze follows the predeclared interpolation plan without family
selection; no parameter adjustment from repeat/validation is permitted. No model
accuracy, autonomous-plan or planner-improvement claim from these training nodes.


Validation runner prepared after freeze: run_validation.sh checks the exact
frozen model SHA, fitting source/fit-plan hashes, both complete31/5 training raw
sessions and their quality gates. It reproduces frozen parameters from train1
before any validation process launches; no train2 fit. Then runs two72-cell
validation sessions with seeds616321/616322, existing PMU/block controls and
model SHA in metadata, exclusive raw/log outputs. Shell syntax, embedded Python
AST and local frozen identities checked. Script and frozen dependencies synced
exec54710; no validation process launched. Full preflight awaits complete train2.
Latest live process check exec73109: train2 PID2555477 present, warmup progress
-4 then-3/31, stderr empty. Existing formal handle60751 remains the training run.


### Matched training repeat evaluated; frozen holdout launched

Remote process inspection exec3acbb7 found no matched collector/runner, train2
stdout ended round31/31 and stderr was empty. Fetch execd5ec9a completed.
Original train2 raw analysis reproduced the remote JSON exactly; evaluator checked
exact protocol,31/5/616312 and frozen implementation/model identity. All measurement
gates passed, elapsed1925.612887s; all56 nodes supported at independent BG-only
read pressures224.318402 and275.530539GB/s. No parameters refitted.

Frozen history/pressure model versus pooled-history control, node competition-delta
MAE in us: full48 nodes2.919037 versus5.657071; tail8 nodes2.551485 versus14.532625;
all56 nodes2.866530 versus6.925008. These are same-condition repeat errors,
conditioned on measured BG-only pressure, not unseen-shape or autonomous-plan
accuracy. Raw/evaluation/audit retained in receiver_tail_matched/train2.*.

Evaluation command: `.venv/bin/python tmp/joint_cost_model_20260911/receiver_tail_matched/evaluate_response.py tmp/joint_cost_model_20260911/receiver_tail_matched/train2.jsonl --model tmp/joint_cost_model_20260911/receiver_tail_matched/frozen_model.json --output tmp/joint_cost_model_20260911/receiver_tail_matched/train2.evaluation.json`.

Started existing `run_validation.sh` on Arm-codex-internal via SSH handle47734,
under existing experiment authorization. Preflight printed both training sessions
validated and frozen parameters/source identities reproduced. Two72-cell sessions,
5warmup/31formal, seeds616321/616322, include M76 and7+7 holdouts; unchanged model
SHA15971f62aa393ae3bdf11f6aba7a744902e87533a08361be5763d87eb8f2d13b.
Validation completion/results remain pending; no planner baseline adoption.


### Matched holdout session1: pressure interpolation overestimates

Remote inspection e5b099 confirmed validation1 ended31/31 with empty stderr,
and validation2 collector PID2753128 running seed616322 under existing runner
handle47734. Fetch b54c71 completed. Local original-raw analyze exactly reproduced
remote validation1.analysis.json; frozen evaluate_response.py completed (dab17f),
checking31/5/616321, protocol and frozen model/source identities. All gates passed;
elapsed2549.103899s, max PMU perturbation0.144147percent, block perturbation
0.215768percent. All70 block-response nodes supported. Artifacts are
receiver_tail_matched/validation1.{jsonl,analysis.json,evaluation.json,audit.json}.

Independent BG-only read rates at6/7/8 teams per LLC:224.594830/257.764661/
275.628115GB/s. Frozen pressure interpolation places7+7 at fraction0.640892
between training6+6 and8+8. Session1 conditional block competition-delta errors:

| Held-out dimension | Nodes | MAE/us | Signed bias/us | Positive errors |
|---|---:|---:|---:|---:|
| M76 history, pressure6/8 |28|5.673672|-5.354252|3|
| pressure7, M52/100 |28|23.737106|23.737106|28|
| M76 history and pressure7 |14|20.219823|20.219823|14|

All42 pressure7 nodes overestimated. W13 tail4 at7+7: M52 predicted217.984515
versus observed165.148us (paired-delta MAD1.474us); M76 200.009365 versus
166.489us (MAD4.197); M100 182.034215 versus130.867us (MAD2.766).
This is systematic held-out pressure error despite good same-condition repeat.
It does not establish whether the missing cause is nonlinear pressure response
or insufficient BG-only average-rate features. History-only transfer also retains
W13 errors (M76 tails -11.587us at6 and -18.495us at8), so it is not declared
fully passed. Node medians are not summed into a claimed whole-call error.

Keep model frozen for the already-running second holdout session; no refit,
family switch, active solver integration, or planner improvement claim. These
are conditional diagnostics using measured background pressure, not autonomous
plan validation. After session2, assess repeatability before changing structure;
any correction makes these points development data and requires new holdouts.


### Matched holdout session2 complete: systematic pressure error repeats

Remote check54a71e showed31/31,empty stderr,no collector. Fetch05522a exited0;
local raw analyze reproduced remote analysis exactly (6e2ea8); frozen evaluator
48927f exited0. Exact31/5/616322 protocol and frozen model identity validated.
All gates passed,elapsed2549.140713s,max PMU perturbation0.159483percent,
block perturbation0.120705percent; all70 nodes supported. Remote process check
eb9bf8 exit1 found neither collector nor validation runner. Old SSH handle47734
had not yet returned terminal output at216b4e; remote actual work is terminal.

Second-session conditional block increment MAE/bias(us): history-only28 nodes
3.150482/-1.220193; pressure-only28 nodes25.190372/+25.190372; simultaneous
history/pressure14 nodes26.238353/+26.238353. All42 pressure7 nodes again
overestimated. W13 pressure7 tail predicted/observed(us): M52 218.977966/162.463;
M76 200.941213/136.534; M100 182.904459/127.530. Thus the frozen pressure-linear
interpolator's systematic held-out error repeats; do not integrate/adopt it.

M76 W13 tail also shows context variation across sessions: solo service medians
437.929->439.989us, but joint6/7/8 medians523.789->502.094,603.701->576.468,
725.932->711.888us. W2 solo212.226->212.351 and joint medians remain much
closer. These are separate service medians, not paired-delta medians, and must
not be substituted into the primary paired metric. W13 variation principally
appears on the joint side; the specific cause is unproven.

Artifacts validation2.{jsonl,analysis.json,evaluation.json}, holdout_comparison.json
retain raw identity, both-session grouped errors and M76 service/MAD diagnostics.
Evaluation used the same command as validation1 with validation2 input/output.
No parameters, production defaults or active solver changed. Next work must
distinguish nonlinear pressure response from inadequacy of BG-only average
pressure, while retaining history/context uncertainty. Any model developed from
these failures requires new held-out pressure/background conditions. Full-plan
and equal-budget planner acceptance remain open.


### Post-hoc pressure-coordinate diagnostic

After both frozen holdouts failed, evaluated q(p)=p/(295-p) using the previously
recorded295GB/s independent read reference (not a universal capacity bound).
Preserved every training response endpoint, history interpolation and original
pressure support; no capacity optimization or response fitting. Revalidated both
original raw sessions through the frozen evaluator before calculation. Explicit
endpoint/zero equivalence checks passed for every supported block/history.
Saved receiver_tail_matched/queue_coordinate_diagnostic.json with both raw/model
identities, individual predictions and grouped errors. Local analysis exec5ab1f9
exited0. This is post-hoc diagnosis, not new held-out acceptance.

| Group | Linear MAE session1/2 us | Queue-coordinate MAE session1/2 us |
|---|---:|---:|
| pressure only,28 nodes |23.737106/25.190372|5.574039/4.552495|
| history and pressure,14 nodes |20.219823/26.238353|9.184247/3.421087|
| history only,28 nodes |5.673672/3.150482|7.276070/4.253235|
| pressure7 full,36 nodes |21.485403/23.918613|5.455684/3.385117|
| pressure7 tail,6 nodes |29.040331/35.266216|14.707988/8.916813|

The curvature diagnostic removes much of the systematic pressure7 overestimate,
but pressure-only bias becomes -5.494178/-4.061185us and tails retain negative
bias. History-only error worsens near the upper endpoint. This does not prove
a queue mechanism, sufficient average-pressure features or autonomous feedback.
Do not replace the frozen model or active solver.

Next discriminating experiment design: freeze this coordinate candidate as one
comparator, collect new intermediate pressure conditions without fitting their
foreground residuals; separately compare different background compositions at
matched independently measured BG-only pressure and locality. Select matching
conditions using BG-only measurements, then freeze selections before foreground
measurement. Preserve same frontend/window/history/probe protocol and paired
solo controls. If composition differences exceed repeated-session uncertainty,
scalar average pressure is insufficient even with curvature. Use both stages and
full/tail groups; retain M76 W13 context variation as an unresolved limitation.
No additional remote collection launched by this diagnostic. Full-plan and
equal-budget planner acceptance remain required and incomplete.


### Background-composition discrimination prepared

Prepared ignored receiver_pressure_composition/candidate.json (frozen parent
responses, q=p/(295-p), no capacity tuning) and design.json. No runtime code or
remote run yet. Native background already supports M12/48/96,W13/W2,copies32;
current matched driver hardcodes M48/W13 and needs an isolated generalized driver
and matching complete-grid analyzer. Screening24 BG-only conditions:3M x2stage
x4 symmetric team counts5/6/7/8; paired PMU plus idle gives50 cells per round.
5warmup/31formal,two seeds616401/616402. Initial arithmetic assertion mistakenly
expected48 and failed; corrected design/count to24 before any measurement.
Uniqueness, parent identity and disjoint CPU placement checks passed.

Predeclared matching uses BG-only data from both sessions: same team count and
copies, different(M,stage), read mismatch<=2percent, write mismatch<=2GB/s,
within calibrated read domain; choose up to two disjoint best pairs. No matched
pair means no matched-pressure claim, not relaxed thresholds. Separately select
new eligible conditions near quarter/three-quarter calibrated pressure range,
excluding original M48/W13 anchors; selections freeze before foreground timing.
Foreground M52/100 and both stages retain solo,PMU and block-timing controls.
Screening measurement floor30min plus setup; full-operation estimate, smoke and
resource-authorization scope check required before launch. Candidate is conditional
and post-hoc, not adopted. Dynamic joint-runtime and full planner gates remain open.


### Composition selector implemented; collection remains pending

M-class Lab selection.py consumes typed BG-only observations with complete24-cell
x2-session grid and31 finite read/write samples. Rejects missing/duplicate cells
and failed quality gates; original-domain eligibility required in both sessions.
Matching implements predeclared2percent read/2GB/s write thresholds, same team
count, deterministic greedy disjoint pairs, quarter/three-quarter new-pressure
targets and original-anchor exclusions. Explicitly defines relative denominator
as max(a,b) and target ranking by worst-session distance before any screening.
Caller must authenticate original raw protocol and native/PMU identities; this
module is not a stand-alone raw validator or measurement pipeline.

Command `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_pressure_composition/test_selection.py`:10 passed0.13s. Tests cover shuffled-input
determinism, second-session mismatch, write/team mismatch, empty support,
missing/duplicate/NaN/short/failed gates, and original-anchor exclusion. Ruff
format/check passed. Source review found no active solver/native/production
callers; math selection contract and manifest status synchronized. No remote
collection or performance claim. Next implement BG-only collector and raw
analyzer, then smoke and whole-operation resource check before screening.

### BG-only collector and raw validation prepared

Implemented receiver_pressure_composition/measure.py and analyze.py using the
existing native handshake and paired PMU protocol:24 BG conditions plus idle,
PMU off/on gives50 cells, two BG processes and no FG process. Raw validation
checks frozen protocol/candidate/native identities, exact259 counter-name
coverage without duplicates, complete grid, CPU/numerical status, positive
integer team calls, finite service/window/counter values and session duration.
Signed idle-corrected demand remains unclipped. Quality gates preserve records
while rejecting count resolution above5percent and paired per-role PMU median
perturbation above5percent.

Command: .venv/bin/python -m pytest -q
tmp/joint_cost_model_20260911/receiver_pressure_composition/test_protocol.py
tmp/joint_cost_model_20260911/receiver_pressure_composition/test_selection.py
Result:27 passed in0.25s. Complete50-cell synthetic fixture checks signed budgets;
negative coverage includes missing/duplicate cells, wrong events and identities,
NaN/empty calls, infinite service, failed CPU check, multiplexed counters and
invalid duration. Independent quality failures remain visible. Ruff format/check
passed for the three touched Python files. This is local L1 evidence only;
event encodings, persistent native integration and hardware perturbation require
Arm smoke. Authenticated two-session raw-to-selector reduction is still pending.
No remote collection, active response-equation change or performance claim.

### Two-session raw-to-selection integration

Added receiver_pressure_composition/freeze_selection.py. It accepts exactly two
raw JSONL byte streams, independently runs analyze(), requires predeclared
seeds616401/616402 and5/31 rounds, and rejects duplicate raw inputs, duplicate
seeds, incomplete cells and failed quality gates. Calibration bounds come from
the frozen candidate. All48 condition/session observations, including signed
writes and ineligible cells, remain in the output. Deterministic selection uses
only BG observations; no-match is explicit. CLI output uses exclusive creation.
Raw, candidate, protocol/design and implementation hashes provide provenance
traceability, not independent hardware attestation.

Command .venv/bin/python -m pytest -q with receiver_pressure_composition/
test_protocol.py test_selection.py test_freeze_selection.py:35 passed in3.57s.
Eight new integration tests consume complete synthetic two-session5/31 raw
records through the real analyzer, including input reversal, duplicate/incorrect
sessions, late missing cells, gate failure and no-match. Ruff check passed.
No hardware result or adoption claim; Arm smoke and fresh measurement are next.

### First Arm screening smoke exposed fixture/schema mismatches

Executed on Arm-codex-internal at /home/zhangxu/codex/fused_cpp:
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
numactl --physcpubind=240-319 --membind=3 .venv/bin/python
tmp/joint_cost_model_20260911/receiver_pressure_composition/measure.py
--split screen --rounds1 --warmup0 --seed616400 --output
tmp/joint_cost_model_20260911/receiver_pressure_composition/smoke.jsonl
(CLI flags were passed with separated values.) Native collection completed all50
cells in33.61743663s. Raw numerical/CPU checks passed, but analysis failed:
the synthetic fixture incorrectly simplified DDRC names to ddrc0..7; actual
devices use ddrc{0,2,3,5}_{0,1}. After correcting coverage, analysis exposed a
second fixture error: unchanged background_native emits median_ns and
team_median_ns, not the foreground mean_ns/max_ns schema. Neither failed
analysis is acceptance evidence. Original smoke.jsonl is preserved unchanged.

Before any formal acquisition, protocol/design v2 explicitly changes the BG
instrumentation gate to paired median-service perturbation, retaining5percent.
The analyzer now consumes median_ns; selector thresholds and native binary
remain unchanged. Revised35 tests passed in3.63s and Ruff checks passed.
A separate smoke_v2.jsonl run uses the same50-cell command and seed. This change
corrects unavailable fields, not a threshold adjustment based on response error.

Second smoke completed50 cells in33.63962958s, remote analyzer exit0. Local
raw reanalysis exactly equals the full remote analysis after JSON normalization.
Maximum absolute paired BG median-service PMU perturbation1.19776212percent;
maximum single-call fraction0.0052083333 (0.520833percent); both gates pass.
Artifacts: smoke_v2.jsonl, smoke_v2.analysis.json, smoke_v2.audit.json.
Raw SHA256 ce9ad50eff80979352b910d5d9d015e002debf57797f4480573f6f6edec7b0a1.
No new native build: existing background_native SHA
9a7064b3b6fdbb59b1a0f9f990cee78f784a03d06d04274d0f990e2c024c2901
matched before execution. Actual geometry remains4T full owner stripes, H4096,
F512, W13/W2 full8/4MiB, per-thread2/1MiB, copies32, NUMA3 local/cross
CPU end316/276; foreground CPU319 reserved but no FG process launched.

Screening two-session5/31 wall-clock projection from this smoke is40.37min,
not the30min observation-only floor. Foreground follow-up can contain at most
126 cells/round (four FG shape/stage combinations x7 conditions x4 controls,
plus7 BG-only conditions x2 PMU controls);72 total warmup/formal rounds across
two sessions give75.6min observation floor. Prior matched FG sessions show
roughly1second/cell including setup, so allow about150min for that stage and
about210min total including screening and margin, conditional on selected cells.
This is an estimate, not authorization or a launched formal run. Foreground
driver/evaluator implementation and integration remain to be completed before
the whole-operation resource approval check. Full-plan and planner gates remain.

### Foreground composition collector and raw analyzer implemented

Added followup_protocol.py, measure_followup.py and analyze_followup.py under
receiver_pressure_composition. Before any native launch, the CLI revalidates
both original screening JSONL sessions and exactly compares the recomputed
selection artifact; empty selection rejects foreground work. Cell identity
explicitly separates foreground M/stage from background M/stage/team count.
K selected conditions produce18(K+1) cells, maximum126, retaining same-session
BG-only pressure, solo FG, PMU on/off and block timing on/off. Foreground-first
ARMED/GO ordering matches the existing matched collector; native binaries and
CPU/weight geometry unchanged.

Raw analyzer binds selected conditions, selection/candidate/protocol/native
identities, complete grid, actual DDRC names, counters and native observations.
FG mean and BG median services have separate validation and PMU controls;
FG block sums plus overhead must reconstruct mean service. Count resolution,
paired PMU and FG block-timing gates remain5percent. It does not compare response
predictions or fit coefficients.

Command .venv/bin/python -m pytest -q with receiver_pressure_composition/
test_protocol.py test_selection.py test_freeze_selection.py
test_followup_protocol.py test_followup_analysis.py:50 passed in5.10s.
Fifteen new tests cover126-cell control closure and commands, disjoint placement,
same-session BG-only presence, foreground-first launch order, invalid conditions,
tampered selection against actual synthetic raw reanalysis, separate native
schemas, selection binding, missing cells, block accounting and independent
instrumentation failures. Ruff checks passed for five new Python files.
measure_followup.py --help imports and exits successfully on local macOS.
No remote foreground execution or formal acquisition this turn. Next implement
frozen linear/queue response and matched-composition comparison, then hardware
integration and whole-operation resource approval check. Full-plan/planner
acceptance remains open.

### Frozen response and composition evaluator implemented

evaluate_followup.py accepts formal foreground raw data only (5/31, seeds616411
or616412), validates it through analyze_followup, and CLI-revalidates frozen
selection from original screening. Predictions use same-session BG-only median
read pressure. Actual block increments use paired-round PMU-off/block-timing-on
joint-minus-solo times, preserving signed samples and MAD. Linear and fixed295
queue-coordinate models share unchanged parent responses/history and original
support. Out-of-domain observations remain explicit with supported counts;
no extrapolation or fitting. Aggregate MAE/bias separates stage and full/tail.

Selected composition pairs are checked again against current-session read/write
pressure and original domain. Paired-round observed difference, predicted
difference and residual difference are preserved even when matching fails.
A failed match cannot support matched-pressure causality, and single-session
MAD is not a significance or repeatability test. Cross-session analysis remains.

Focused command .venv/bin/python -m pytest -q
tmp/joint_cost_model_20260911/receiver_pressure_composition/test_evaluate_followup.py:
4 passed in1.51s; Ruff passed. Tests verify every supported frozen endpoint and
zero response, intermediate curvature, no extrapolation, full raw formal
evaluation, retention of unsupported points, and matching drift from240 to250GB/s.
No new foreground hardware data, response parameters or planner adoption.

### Formal pipeline prepared; resource approval pending

Added run_formal.sh (M-class Lab orchestration only). Sequence:screen1/2 at5/31,
analyze each, freeze selection from both raw sessions, one-round foreground
smoke on actual selected conditions and gate it, foreground1/2 at5/31, analyze
and evaluate each. Missing selection or any failed command/gate stops the
pipeline. Exclusive formal/ directory and exclusive data outputs prevent
automatic restart or overwriting partial evidence. GNU timeout210m with15s
kill grace bounds the pipeline process group; no auto-retry. Source hashes and
console log stay under formal/. COMPLETE means collection/per-session steps
finished, not cross-session interpretation or model acceptance.

Remote read-only preflight found no live receiver measurement/native processes
apart from the inspection command itself. Existing FG/BG SHA identities still
match frozen protocol. Synced only owned Python and pipeline sources.
Command on Arm-codex-internal:/home/zhangxu/codex/fused_cpp:
bash -n tmp/joint_cost_model_20260911/receiver_pressure_composition/run_formal.sh;
timeout --version reports GNU coreutils9.0;
.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_pressure_composition
=>54 passed in15.58s; evaluate_followup.py --help exit0.
These tests use synthetic raw data, not new foreground hardware integration.

Prepared launch command (NOT EXECUTED):
bash tmp/joint_cost_model_20260911/receiver_pressure_composition/run_formal.sh
Expected whole operation approximately190min, with210min cap plus15s kill grace:
screening about40min from smoke, maximum foreground about150min from prior
matched setup-inclusive timing, plus smoke/control overhead. Fewer selected
conditions reduce duration. Resources:existing Arm machine, NUMA3 CPU240-319,
existing native binaries, no production/default changes. Previous bounded
experiment is complete; approval for this new batch/resource scope is pending.
Do not launch from this preparation record alone.

### Cross-session comparison prepared while resource approval is pending

Added compare_sessions.py and attached it to the local run_formal.sh after both
session evaluations. It reevaluates original raw data, enforces distinct
foreground seeds616411/616412 and same candidate, and preserves individual
session results. Queue MAE lower in both sessions requires complete support
in both; no averaging away unsupported coverage. Composition repeatability
requires matched pressure in both sessions and preserves observed/predicted
differences, residuals and MAD. Same-sign nonzero residuals are descriptive
only, not significance or proof of a hardware cause.

Command .venv/bin/python -m pytest -q
tmp/joint_cost_model_20260911/receiver_pressure_composition/test_compare_sessions.py:
7 passed in2.56s. Three integration tests use complete synthetic raw sessions
through the real evaluator (input reversal, unsupported second session,
duplicate seed). Four isolated tests verify sign/matching logic with controlled
evaluation outputs. Ruff and bash -n run_formal.sh passed. No existing
single-session evaluator or calibrated coefficient changed.

The requested210min budget has not received an explicit answer. No acquisition
was launched or treated as live; this turn completed independent local analysis
work. New comparator and updated local runner still need synchronization before
an authorized launch. Model, dynamic-plan and planner acceptance remain open.

### Final preparation verification; blocked on new batch authorization

Synced compare_sessions.py, test_compare_sessions.py and run_formal.sh to the
existing Arm Lab directory. Remote command:
bash -n tmp/joint_cost_model_20260911/receiver_pressure_composition/run_formal.sh
and .venv/bin/python -m pytest -q
tmp/joint_cost_model_20260911/receiver_pressure_composition
passed61 tests in21.18s. Remote test ! -e receiver_pressure_composition/formal
(with full Lab prefix) succeeded: formal acquisition has not started.
Local/remote exact source SHA256 match:
compare_sessions.py043f21c8db6db8035e9ec69b1bbcbef7a133ddabbab3a3dd66139faf7bdb07ab;
run_formal.shdcf79cd451f170379f25bddf9f83177d8890e1c950dce8a2c04f65227471966c.

The explicit210-minute new-batch request remains unanswered across three
consecutive goal turns. Those turns completed independent comparator and
deployment preparation, but the next discriminating evidence requires the
authorized acquisition. Do not infer authorization from automatic goal
continuations. Goal blocked pending that resource decision, not completed.
On explicit authorization, inspect current remote processes/output state before
launching the prepared script; never restart an existing partial run implicitly.
After collection, inspect both-session errors and composition match/repeatability;
do not adopt the queue comparator based on synthetic tests. Dynamic joint
simulation, unseen full plans and equal-budget planner acceptance remain required.

### User authorized new210-minute batch; formal pipeline launched

User explicitly replied 授权 to the concrete210-minute experiment request.
Authorization now covers this batch on the agreed Arm NUMA3 machine, including
screen1/2, selection freeze, selected-condition foreground smoke, foreground1/2
and frozen comparison, within210min plus15s termination grace. Prior pending
authorization entries are historical; do not ask again within this scope.

Prelaunch remote inspection found no receiver/native/210m processes and no
formal directory; runner/comparator source hashes matched the final prepared
identities. Started the previously prepared command over SSH (exec handle70767).
At2026-09-15 01:45:45 UTC, verified live remote timeout PID/PGID2903692,
screen1 Python2903761 and BG native2903762/2903763, all in process group2903692.
console.log reports Screening session1. No observation failure or restart.
Ignored authorized_run.json records handles, output path and authorization.

Current state is running, not completed or quality-accepted. Poll the existing
SSH handle and process group; inspect formal/console.log and raw records before
deciding terminal state. Never restart based solely on polling timeout or a
missing SSH connection. Full batch/model/planner acceptance remains outstanding.

### Authorized formal composition batch completed; candidate fails fresh condition

Status inspection confirms COMPLETE exists, all planned per-session outputs
and comparison.json exist, and no corresponding measurement/native/timeout
process remains. Total measured stage elapsed78.7165min (screen1201.063/1201.158s,
FG smoke32.085s, FG1144.336/1144.349s), below210min budget.
Both screening sessions, FG smoke and both FG sessions pass all declared gates.
Downloaded full raw artifacts to receiver_pressure_composition/formal/.
Local independent reanalysis of all5 raw sessions, recomputed frozen selection
and cross-session comparison exactly match remote JSON-normalized artifacts.

Screening found5 eligible conditions but zero qualifying composition pairs.
Only new-pressure condition selected:M12/W2,6 four-thread teams per LLC.
FG-session BG-only reads260.444028/259.603497GB/s; writes0.075912/0.068753GB/s.
Both FG sessions cover all28 block-response points within supported pressure.
This is a new pressure/background condition, but NOT a matched-composition
experiment: no claim that scalar pressure suffices or is insufficient follows.

Paired-block competition-increment MAE, linear vs frozen queue, session1/2 us:
W13 full15.954656/13.640640 vs18.570947/21.013443;
W13 tail4 15.782039/12.951183 vs47.162595/50.227685;
W2 full9.636722/6.435651 vs9.286585/12.570825;
W2 tail4 6.787292/5.695236 vs24.773516/26.303924.
Queue errors are negative for every point in both sessions: systematic
underprediction. W2 full gains slightly in session1 only; no group has lower
queue MAE in both sessions. Candidate not adopted. These are response-increment
errors, not complete-GEMM, lane or full-plan absolute prediction errors.
No response/T0 parameters changed. Need a newly designed BG-only matching
experiment to discriminate composition effects from pressure curvature.
Full dynamic model and equal-budget planner acceptance remain outstanding.

### No-pair diagnosis and retrospective configuration contrast

Recomputed all24 screening medians and same-team pair margins. Only same-team
pair with read mismatch<=2percent is M12W13 vsM12W2 at8+8 (1.694percent),
but both are outside calibrated pressure support. Within-support same-team
M12W13/W2 at5+5 differs9.602percent. Writes are not the limiting factor.
M12W13 at6+6 misses the upper bound narrowly in one session; M48W13 at6+6
misses the lower bound in one session. Keep those original exclusions.

M12W2 at6+6 vsM48W13 at7+7 is a close-pressure pair with DIFFERENT task counts.
Revalidated both old matched validation raw sessions against their existing
evaluation JSON, then compared old/new shared28 nodes across all four independent
session combinations, not artificial paired rounds. Saved
cross_composition_retrospective.json with source hashes and all node contrasts.
W13 tail4 M52 old165.148/162.463 vsnew210.190/211.942us;
M100 old130.867/127.530 vsnew178.681/176.186us.
W2 tail4 M52 old79.072/80.577 vsnew113.500/112.247us;
M100 old69.987/71.013 vsnew68.506/70.170us.
Do not conceal the latter history difference through a pooled W2-tail average.

Subtracting frozen linear or queue predicted pressure differences leaves grouped
mean new-minus-old residuals across four cross-session combinations:
W13 full12.27..16.80us, W13 tail36.01..42.99us,
W2 full6.68..10.74us, W2 tail11.54..13.99us.
This is retrospective, with different team counts, dates and acquisition grids;
candidate-based pressure correction is not a causal control. It motivates,
but does not replace, a contemporaneous interleaved configuration contrast.

Prepared next_configuration_contrast_design.json: fixedM12W2/6+6 andM48W13/7+7,
same foreground M52/100,W13/W2 and matched geometry;54 cells including all
PMU/timing/solo/BG-only controls, two5/31 sessions616421/616422. Recheck same
2percent read/2GB/s write matching in both sessions. This explicitly tests
whole configuration differences including team count, not isolated M/stage.
Original no-pair result remains unchanged. Estimated65min plus smoke/analysis,
proposed90min cap for a new batch; no new remote collection or implicit reuse
of old remaining budget. Input/validation implementation remains pending.

### Independent fixed-configuration experiment implemented locally

Created ignored receiver_configuration_contrast using snapshots of the existing
collector/PMU/analysis structure; source_origin.json retains source identities.
The old receiver_pressure_composition files and same-team selection rule were
not changed. New entrypoints accept --design instead of screening selection,
verify exact fixed configuration design, and bind design_sha256 in metadata.
FixedM12W2/6+6 vsM48W13/7+7, formal seeds616421/616422, all54 control cells.
New evaluator permits the intentional unequal-team pair and keeps original
pressure/support/quality gates; no isolated stage/task-count causal claim.

Command .venv/bin/python -m pytest -q
tmp/joint_cost_model_20260911/receiver_configuration_contrast/test_contrast.py:
4 passed in1.63s. Full synthetic raw-to-cross-session validation checks28
paired blocks and a known20ns block difference, pressure matching versus
second-session240->250GB/s drift, design tampering and old-seed rejection.
Ruff check/format passed; measure_followup.py --help exits0 locally.
run_formal.sh syntax validation passed:90min timeout, exclusive formal directory,
54-cell smoke and gates before two5/31 sessions and cross-session comparison.

No remote source synchronization or new measurement this turn. Remote checks
and new bounded-batch resource authorization remain before launch. The predicted
configuration/history mechanism remains unproven; no model parameters or active
solver changed. Full dynamic-plan and equal-budget planner validation remain.

### Fixed-configuration remote preflight complete; new90min authorization pending

Verified the new remote receiver_configuration_contrast directory did not exist,
then synced only its owned Python/JSON/runner files. No active receiver
measurement/native/90m/210m process was found (inspection command excluded).
Existing FG and BG native SHA identities match the frozen protocol; no build.
On Arm-codex-internal:/home/zhangxu/codex/fused_cpp:
bash -n tmp/joint_cost_model_20260911/receiver_configuration_contrast/run_formal.sh;
.venv/bin/python -m pytest -q
tmp/joint_cost_model_20260911/receiver_configuration_contrast/test_contrast.py
=>4 passed in3.48s; measure_followup.py --help exit0.
These are synthetic raw-data checks on Arm, not actual native acquisition.
Remote formal/ output directory remains absent.

Local/remote exact hashes match:
run_formal.shec1ee544cfa395bf209fc53b9ac287d6d54ed03c17fa57ce9630d44cbcc89217;
design.jsone573f9a01480e153de0140f318c459aac00fd2e146483f7942708b6f44f16c21.
Prepared command, NOT EXECUTED:
bash tmp/joint_cost_model_20260911/receiver_configuration_contrast/run_formal.sh
The batch has one54-cell native smoke and quality gate, two independent5/31
sessions616421/616422, raw validation and frozen comparison. Estimated65min
plus smoke/analysis,90min cap with15s kill grace; exclusive formal directory,
no auto-retry. Same Arm NUMA3 CPU240-319, existing binaries and production defaults.

This is a new fixed-configuration experiment following the completed
previous79min batch; the earlier210min authorization is not silently reused as
an open-ended budget. New90min batch authorization requested only after local
and remote preparation completed. No new response evidence or model adoption.

### 2026-09-15：事件驱动续接与固定配置对照启动

用户更新授权：既定机器和 cost-model 研究范围内，预计不超过 4 小时的实验无需逐批确认。此前固定配置对照等待单独 90 分钟授权的状态已被此规则取代。

已接入 `scripts/run_experiment_notify.py`：独立后台进程通过 OS `wait()` 等待实验命令结束，然后一次性使用 `codex queue` 向当前线程投递续接消息。无模型定时轮询；SSH 断线记录为远端状态未知，不能据此重启实验。通知失败保留证据，不自动重发。真实队列通道测试已返回 accepted；实际事件消费仍需在后续 turn 确认。机器和 Codex 队列服务需要持续可用。

固定配置对照 `receiver_configuration_contrast` 于 2026-09-15 15:26:21 UTC 启动，远端 90 分钟上限，NUMA 3 / CPU 240–319；M12/W2 6+6 与 M48/W13 7+7，其余冻结协议不变。已一次性确认远端进入 integration smoke（timeout PID 3370325）。本地 supervisor PID 40078，SSH PID 40079；状态目录 `tmp/joint_cost_model_20260911/receiver_configuration_contrast/notification_run/`，原始结果在远端同实验目录 `formal/`。启动前确认无旧 formal 目录及相关在跑进程；design、FG/BG native 哈希与准备阶段一致。

运行工具验证：`.venv/bin/python -m pytest -q tests/test_run_experiment_notify.py`，7 passed（最终复核 7.74 s）；Ruff format/check 通过。覆盖正常/失败/超时退出、SSH 255、命令启动失败、通知失败、重复 worker 与独立后台 CLI 重复启动。测试使用假队列，不产生额外对话消息。此次仅变更实验运行方式与授权记录，不改变 kernel、成本模型或验收标准；实验完成后仍须复核原始数据和质量门槛。

#### 固定配置对照的结果解读顺序（采集过程中只读复核）

已复核冻结的 `evaluate_followup.py` 与 `compare_sessions.py`。输出保留每个配置、前台 M、stage、history 的逐轮竞争增量，两个正式 seed 分别验证，完整块和 4 行尾块分组；总体分组 MAE 同时混合了两个配置和两个前台 M，不能独自作为泛化结论。结果到达后按以下顺序解读：

1. 先确认两轮质量门槛，以及两配置 BG-only 读压力均位于原 p6–p8 区间、相对差不超过 2%、写压力绝对差不超过 2 GB/s。任一轮不匹配，则此次数据不能当作成功的等压力配置对照；不得追改门槛。
2. 逐个保留 M52/M100 × W13/W2 × history 的两轮结果，尤其单独列出四个尾块节点。配置差使用同轮竞争增量之差的 median，不能替换为两个独立 median 之差；两种统计量一般不相等。
3. 分别扣除冻结 linear/queue 模型对于剩余压力差的预测差，再报告残差大小、逐轮 MAD 和跨轮方向。一致符号仅为描述性复现，不等同于统计显著性。
4. 若存在稳定配置残差，只能说明当前 BG-only 标量压力与冻结响应尚未解释该配置差。因为两配置同时改变 M、stage 和 team 数，不能单独归因于其中之一，也不能直接证明 LLC 或 DRAM 的具体硬件机制。
5. 竞争增量误差仍不等于绝对 GEMM、lane、完整计划误差。是否修改响应特征须根据上述结果决定；无竞争成本保持冻结，后续动态完整计划和等预算 planner 验收仍独立保留。

此次复核没有修改正在运行的实验脚本、冻结候选或分析程序。

#### 条件响应到自主动态模拟的接口审计（2026-09-15）

只读核对 `receiver_configuration_contrast/evaluate_followup.py`、`dynamic_resource_candidate/adapter.py` 和 `simulate.py` 后，确认目前不能把新表直接塞进动态模拟器并声称等价。存在三个具体接口差异：

- **压力定义不同。** 条件表输入为同场 BG-only 的 read GB/s，即没有前台参与时的背景服务率；动态 queue 分支则把其他 active expert 的实际 read+write KiB/us（独立请求率乘预测速度）作为压力，且该速度受前台反向影响。二者不仅单位不同，还改变了是否包括写请求及是否包含前台反馈。单位换算不能消除这个差异。自主接入须明确使用哪一种定义，并通过背景速率留出检验；不得把联合总 DDRC 或实测前台进度偷偷作为预测输入。
- **响应作用位置不同。** 新表估计每块 wall-time 增量 `Delta(s,h,p)`；独立时钟模拟器的 queue 项改变 memory-service 速度，计算时钟独立前进。恒压、同起点的简化极限为 `T=max(T0,L/v)`。因此直接令 `v=T0/(T0+Delta)` 仅在 `L=T0` 等额外假设下等价；请求窗口 L 尚未独立确定时，不能由单个增量反推出唯一 memory-service 响应。这需要显式的嵌入假设及恒压还原检查。
- **适用粒度不同。** 新表仅覆盖 1T、完整块 h0..7、4 行尾块 h4..8；需求 adapter 的节点来自独立 M 请求预算，其 prefix 差分不是实测逐块请求。它支持的较宽 team/M 范围不能扩充响应表的已验证范围。M1–11 仍要求显式的未验证 M12 anchor 策略，不能算作泛化通过。

下一次接入分开验收：先做受支持节点的恒压响应还原，再对留出动态场景检验背景速率及阶段重叠预测，最后看固定计划绝对完成时间和等预算 planner 选择效果。此次固定配置对照只回答第一层条件响应的配置依赖问题，不会自行补齐后两层。未改任何模型代码、系数、请求窗口或生产默认。

### 固定配置同场对照完成（2026-09-15）

`receiver_configuration_contrast/run_formal.sh` 正常退出，supervisor 于 16:28:19 UTC 记录 code 0，16:28:20 UTC 通知队列 accepted；距启动约 62 分钟，在 90 分钟上限内。原始数据及 COMPLETE 已同步至本地同实验目录 `formal/`。smoke 51.150 s，两轮正式采集 1830.029/1830.126 s。三份 raw analysis、两份 evaluation 和 comparison 均从原始 JSONL 独立重算，与远端 JSON 完全一致（Python tuple 经 JSON 序列化为 list 后比较）；记录于 `formal/local_parity.json`。PMU、完整块计时和计数分辨率门槛两轮均通过，所有 56 个条件响应点/轮在原支持区间内。

配置 A=M12/W2 6+6，B=M48/W13 7+7；BG-only read 为 A 263.508/263.513、B 258.299/258.435 GB/s；相对差 1.9765%/1.9270%，均通过冻结 2% 门槛，但接近边界，不能当成压力完全相等。write 为 A 0.07174/0.08301、B 0.14526/0.14261 GB/s。全部 28 个配置比较节点在两轮均满足匹配要求。

竞争增量 MAE（us，第一轮/第二轮；不是绝对 GEMM 或全计划误差）：

| 前台分组 | 冻结 linear | 冻结 queue 坐标 |
|---|---:|---:|
| W13 完整块 | 26.889 / 26.160 | 7.254 / 7.973 |
| W13 尾4行 | 57.341 / 55.965 | 4.623 / 5.922 |
| W2 完整块 | 15.729 / 14.562 | 3.201 / 4.462 |
| W2 尾4行 | 12.898 / 11.954 | 18.092 / 19.803 |

queue 对 W13 两类及 W2 完整块的改善在两配置、两轮分别成立；不能因此忽略 W2 尾块。A 配置 W2 尾块 queue MAE 24.356/26.709 us，B 为 11.828/12.897 us。A 的 M52 W2 尾块实测竞争增量 99.812/103.001 us，预测 81.775/81.789；M100 则实测 92.214/93.753，预测 61.539/61.546。

扣除 queue 对剩余压力差的预测后，同轮 A−B 残差（us）：

| 节点 | 第一轮 | 第二轮 |
|---|---:|---:|
| M52 W13 h4 尾块 | 0.382 | -5.353 |
| M100 W13 h8 尾块 | -0.489 | 2.459 |
| M52 W2 h4 尾块 | 9.651 | 11.110 |
| M100 W2 h8 尾块 | 13.712 | 17.033 |

W13 尾块残差不保持方向，不能沿用旧跨日期诊断认定其存在大幅稳定配置残差。W2 两个尾块均保留同向配置残差；M100 的逐轮差 MAD 9.915/9.276 us，离散程度须保留。符号重复是描述性证据，不是显著性或 LLC/DRAM 机制证明。两配置同时改变 stage、M 和 team 数，不能分离归因。

本次还暴露跨批次变化：旧 A 配置 W13 尾块增量约 210/212（M52）、179/176（M100）us，本次降至 185/182、147/150；M100 W2 尾块由旧约 69/70 升至 92/94 us。压力变化本身不能被忽略，后续须使用原 raw 在统一模型和压力口径下分解该漂移，不能仅凭新批次改善推翻旧泛化反例。

决策：不采用 queue 为通用响应，不改 T0 或生产默认。下一步优先分解 W2 尾块的条件配置残差与跨批次漂移，再确定需要补充的历史/竞争特征；动态压力、完整计划和等预算 planner 验收仍未完成。

### 同一 A 配置的跨批次漂移分解（2026-09-15）

对旧 `receiver_pressure_composition/formal/foreground{1,2}` 与新 `receiver_configuration_contrast/formal/session{1,2}` 的 A=M12/W2 6+6 配置，核对每份 evaluation 绑定原始 JSONL SHA256、冻结 candidate 身份、FG/BG native 身份及 500ms 窗口一致。采集网格不同（旧一个背景配置、新两个），不是环境完全一致的重复实验。全部 28 节点保留 old/new 两场和四种跨场组合，不将跨日期轮号伪配对。产物 `receiver_configuration_contrast/cross_batch_a_drift.json` 保留输入哈希及独立统计量。

下表为 new−old、四种跨场组合范围，单位 us。无竞争/联合列分别是各自 median 的差；竞争增量为同轮 joint−solo 后取 median 的差，不能要求三列严格相减闭合。

| 尾块节点 | 无竞争时间变化 | 联合时间变化 | 竞争增量变化 | 扣除冻结 queue 压力差后的残差 |
|---|---:|---:|---:|---:|
| M52/W13 h4 | -1.048 至 +0.509 | -29.490 至 -25.270 | -30.184 至 -24.934 | -48.886 至 -39.943 |
| M52/W2 h4 | -0.367 至 +1.758 | -13.308 至 -8.351 | -13.688 至 -9.246 | -21.192 至 -17.551 |
| M100/W13 h8 | -0.651 至 +0.767 | -31.710 至 -26.239 | -31.796 至 -25.920 | -45.658 至 -41.588 |
| M100/W2 h8 | -0.363 至 +1.064 | +21.943 至 +26.048 | +22.044 至 +25.247 | +17.296 至 +21.423 |

结论限于这四个尾块：漂移不由无竞争基线的同等幅度变化解释。新 A 的 BG-only 读率比旧 A 更高，但三个节点的竞争增量反而降低，因此冻结单调压力曲线对这些节点不仅不能消除漂移，还扩大了残差。M52/W2 与 M100/W2 方向相反，不能用全局时间倍率或单一批次压力偏移一起纠正。这个观察尚不能区分采集顺序、进程/分配状态、历史响应或其他竞争状态，不能单独归为硬件频率或缓存驻留。

后续首先利用既有 raw 的执行顺序检查前序条件及轮内位置是否关联响应变化，并核对两网格的准备/复用行为；若需要新的辨识实验，应保持 A 配置与 native 固定，只控制网格上下文/准备状态。不得把批次编号拟合为 planner 特征。模型、T0 和验收门槛均未改变。

### 前序条件关联与准备行为审计（2026-09-15）

逐行解析两批各两场 raw 的实际执行顺序，保留正式轮目标 A 单元、紧前单元（包括跨轮边界）、轮内位置及同轮 solo 扣除后的尾块增量。全部 16 个 batch/session/M/stage 分组及原始观察保存在 `receiver_configuration_contrast/predecessor_diagnostic.json`，包含输入 SHA256。以下为事后描述性分层，不是预注册因果检验，未拟合参数。

新批次 M100/W2 尾块的前序分层（竞争增量 median us；括号为样本数）：

| 紧前测试的背景 | 第一轮 | 第二轮 |
|---|---:|---:|
| A=M12/W2 6+6 | 70.19 (12) | 61.90 (9) |
| B=M48/W13 7+7 | 96.20 (8) | 94.59 (16) |
| 无背景（idle 或 solo） | 92.79 (11) | 93.23 (6) |

这组方向在两轮重复，量级足以影响整体中位数。旧批次在前序 A 层为 67.91/67.83 us，与新批次同层较接近；旧无背景层为 69.53/84.85 us，不应遗漏其漂移。前序 A 的占比/其他类别组成也不同，因此旧新总体 median 的差不能全被称为固定 A 响应改变。前序分类仍合并了前台 stage、PMU、timing 等多个因素，不能唯一归因。

新 W13 也存在前序关联：M52 前序 A/B 为 188.55/171.45 和 193.07/179.70 us；M100 为 153.15/146.05 和 160.66/147.95 us，方向两轮一致。M52/W2 前序 A/B 为 98.24/100.58 和 102.85/103.72 us，关联较小。新四个尾块轮号相关系数范围 -0.249..0.196、轮内位置 -0.165..0.314；这些简单相关不足以认定持续热漂移，也不能排除更复杂顺序影响。

源代码准备行为：FG/BG native 的 A/B/C 大数组在各自进程启动时一次分配，整场复用；每个 active cell 填充输出、重新创建 worker 与 TeamState，copy iteration 重置，并先执行至少 64 次到 ARMED。GO 前 worker 已在运行，之后连续运行到 ACK。采集器为每轮重新 shuffle 全网格，三个 native 进程跨所有 cell 存活。此路径没有逐 cell 的显式权重 cache scrub。故不能描述为“每个 cell 全新分配/完全冷启动”；但 64 次 lead-in 也使直接把差异解释为第一次冷 miss 缺乏证据。进程级分配跨批次不同，单元级复用与历史仍可能影响响应。

决策：下一辨识实验应在同一进程/分配、相同目标 A、相同 PMU/timing 下，成对控制目标前的指定前序单元并平衡执行顺序；将前序 A、B、无背景的差异与默认随机网格分开记录。首先复现 M100/W2 大差异，保留 M52/W2 与 W13 控制。尚不把“前序测试编号”加入 planner，也不把关联直接当作 cache 驻留状态；真实运行需要可计算的物理/历史特征。

### 前序前台 stage 混杂复核（2026-09-15）

将新批次前序背景与紧前单元的前台 stage 交叉分层，四个目标尾块的全部分层、样本数、joint/solo/delta 中位数及逐条观察保存于 `predecessor_fg_stage_diagnostic.json`。这是事后稀疏分层，不据此给出因果显著性。

M100/W2 目标的竞争增量中位数（us）：

| 前序前台 | 第一轮：前序A / 前序B | 第二轮：前序A / 前序B |
|---|---:|---:|
| W13 | 62.043 (n3) / 95.734 (n3) | 59.530 (n3) / 96.120 (n10) |
| W2 | 75.745 (n8) / 96.134 (n3) | 61.600 (n4) / 94.336 (n5) |

在两种前序前台 stage 内，前序 A/B 的方向仍一致；因此“仅仅是前序 W13/W2 比例不同”不足以解释观察。对应 solo 中位数第一轮约212.3..212.8、第二轮213.3..213.6 us，联合时间保留同等方向的大差异，不能归因于被减去的 solo 中位数差。样本较少、前序 M/PMU/timing 等因素仍混合，不能把分层结果作为已辨识的历史响应公式。

下一对照应固定前序单元的前台部分（或全部前序均无前台），只改变背景预处理 A/B/idle；对每个目标分别测 joint 与同样预处理后的 solo，成对次序平衡。该设置能隔离背景预处理效应，但背景预处理后重新启动前台的 lead-in 与真实连续 expert 历史仍不同，后续需要另外验证迁移。

### 受控前序实验实现中（2026-09-15）

新增独立 Lab `receiver_precondition_control`，复制已验证采集依赖并记录来源哈希；不修改原实验或 native。冻结 A/B/idle 三种无前台预处理，每次预处理后测同一 A 目标或 solo。四种前台 M52/M100 × W13/W2、计时/PMU 双开关及 BG-only/idle 控制全部保留。每轮108目标+108预处理窗口，joint/solo相邻且各自重复预处理，顺序相邻轮反转，两正式种子616431/616432再反转；31轮的单场先后数差至多1。

已实现确定性 schedule、复用进程的采集入口和按预处理分别调用既有 raw quality analyzer 的 wrapper；验证 wrapper 还检查实际序列、每次预处理 native/CPU/调用分辨率与窗口记录。三项 schedule 检查通过（0.18s），新增文件 Ruff 检查通过。尚未完成全 raw 合成集成、采集入口审查、远端 smoke 或正式采集，不能把这些检查称为实验可用性或因果结论。

整批计划上限210分钟，预计在用户已有≤4小时授权内。下一步完成上述集成与远端验证后再启动；模型、T0、生产默认保持冻结。

### 受控前序实验集成通过并启动（2026-09-15）

新增合成 full-raw 集成覆盖108目标/轮、三个context单独质量检查与84个响应节点；验证112个跨context/跨场对照，拒绝错序、错误预处理、不足窗口、CPU验证失败、缺行、candidate错配和目标role丢失。正式/烟测 seed、轮数及warmup组合强制冻结。目标 native 不变，新增记录每个预处理 elapsed_seconds，idle窗口也可审计。

本地 `pytest -q .../test_schedule.py .../test_control.py`：12 passed in13.20s；远端同命令12 passed in30.82s。新增Python文件Ruff通过，runner bash -n与CLI help通过。design SHA256 `a86ea70406519f4d38c74fcd695e679d70d3d9def08f609c22e164772d763556`，runner SHA256 `bc3416669527b7f07ea684dd30c7ee479505e6321b1db8ee368154d8b486bcff`，远端一致。

按用户已有≤4h授权，210分钟 fail-fast 流程已通过事件通知启动器启动；本地supervisor PID46135，状态 `receiver_precondition_control/notification_run/`。先跑真实native smoke并经完整检查，再运行两场5/31正式采集，最终compare_control重新校验两份raw并写COMPLETE；任何一步失败保留原目录，不自动重启。远端进程就绪检查已完成。运行中尚无正式结果或模型验收结论；T0、模型和planner默认保持冻结。

### 受控前序结果的压力混杂解读约束（2026-09-15）

在正式结果到达前只读审计 `receiver_precondition_control/analyze_control.py`、`analyze_followup.py` 和 `compare_control.py`。当前 compare 输出的 B−A / idle−A 是相同轮号的 joint−solo 增量之差，尚未扣除 context 间背景服务率变化；通过完整性和 instrumentation 门槛不等于通过等压力检验。三个 context 的原始 PMU 派生记录已分别保留在 `analyses[context].records`，含 BG-only、solo、joint 的全局读写率与按 role 的每调用 core 事件，因此无需为这一诊断更改正在运行的采集。

正式解读须同时报告每场、每个 context 的 BG-only 读写率及逐轮分布，再给出各前台 M/stage/history 的竞争增量和 context 差。若前序改变 BG-only 服务率，不能把未校正的增量差全部称为“相同压力下的历史敏感度”；只有位于冻结响应支持区间时，才可另外展示冻结压力曲线的解释量和剩余残差，且不能把该曲线当作已验收的真值。超出支持区间应明确记为无法用该冻结曲线校正，不外推、不追改采集门槛。

joint DDRC 是全局总量，不能分离前台与背景请求；各 role 的 L2 事件也不能直接等价为该 role 的 DRAM 字节。块时间来自 PMU-off/timing-on 窗口，PMU 率来自另一控制窗口，不能声称两者是同一次执行的瞬时联合观测。同轮号的 context 差是平衡随机顺序中的配对对照，不是同步执行。即便两场重复出现差异，当前实验首先支持受控前序协议下的描述性效应，仍需后续连续运行场景验证才能作为 planner 可计算的历史状态。

本次仅补充结果解读约束；没有修改冻结实验、质量门槛、模型或生产默认，也没有提前声称正式实验通过。

受控前序实验启动后只读检查：远端原 timeout PID3532333 在 elapsed 13:33 时仍存活，`formal/console.log` 已记录 smoke `round 1/1` 和 `all_gates_passed True`，随后出现第一场正式采集的 warmup 轮日志（至 `round -1/31`）。这证明真实 native smoke 已通过 runner 检查且流程进入正式采集；尚未取得两场正式 raw 的完整性/独立重算证据，不能据此声称正式质量门槛或模型验收通过。检查未修改或重启远端流程。


### 受控前序第一场独立验收（2026-09-15，第二场尚未验收）

从远端下载已关闭的 `formal/session1.jsonl`、`session1.analysis.json` 和 `source.sha256`，本地调用冻结 `analyze_control(raw, design)` 独立重算。17份源码/协议哈希全部匹配，JSON规范化后与远端分析完全一致；A/B/idle三个context的块计时扰动、计数分辨率、PMU扰动门槛全部通过，84个响应点。证据 `formal/session1.local_parity.json`，raw SHA256 `a89bd061f62276ea08c9a99e95d9589cfc8a4cdf354b935eedb8227b1dacdd57`。本次未修改远端流程或冻结模型。

这里A/B/idle指目标窗口之前的BG-only预处理；目标joint窗口始终使用A=M12/W2 6+6。以下为第一场31轮的尾4行竞争增量（同轮joint−solo，us）：

| 前台 | 前序A中位数 | 前序B中位数 | 前序idle中位数 | 同轮B−A差中位数 |
|---|---:|---:|---:|---:|
| M52/W13 h4 | 188.464 | 174.350 | 186.301 | -14.102 |
| M52/W2 h4 | 95.633 | 100.121 | 98.307 | 4.805 |
| M100/W13 h8 | 163.667 | 145.159 | 157.323 | -17.264 |
| M100/W2 h8 | 51.397 | 91.798 | 63.056 | 40.112 |

差的中位数不等于中位数之差。M100/W2的增量MAD为A 0.878、B 1.218、idle 9.774 us，提示idle不是明确的缓存状态重置。对应目标A的BG-only读取率中位数，前序A/B/idle分别为262.981852/265.523790/263.367348 GB/s，写入率0.064207/0.066712/0.069128 GB/s（各31点）。因此前序也改变服务率，表中时间差尚未扣除压力变化，不能全部解释为历史敏感度。不同前台节点的效应方向和幅度不同，目前不支持统一乘数修正。

这是单场描述性证据；第二场完整性、可重复性、两场联合比较及连续真实trace迁移尚未验收。不得据此接受历史模型或宣称planner改善。下一步仍为完成第二场独立重算，并按既定压力混杂约束分析两场结果。


第一场压力分布补查：`formal/session1.pressure_distribution.json` 保存分析输入哈希、31点分布和同轮差。A/B读取率MAD分别0.085/0.107 GB/s，范围262.721–263.418和265.057–265.747 GB/s，样本范围不重叠；同轮B−A读取率差中位数2.558 GB/s、MAD 0.123，31轮全部为正。因此约1%的压力偏移不是由少数异常轮抬高中位数造成，不能按测量噪声忽略。idle范围262.714–265.532 GB/s，P10–P90为263.087–264.925，波动更宽。这里仍为不同控制窗口的BG-only服务率，不能据此逐轮归因目标窗口的块耗时；尤其不能用单场均值或中位数相近宣称条件等压。


### 受控前序两场完成并独立验收（2026-09-15）

原supervisor退出事件确认19:58:55 UTC code0，通知accepted；从16:45:10启动至完成约194分钟，低于210分钟上限。同步全部formal数据及COMPLETE；使用冻结 `analyze_control` 重算smoke和两场，`compare_control.compare` 从两份raw重算112个对照，JSON规范化后全部与远端产物一致。17份源码/协议哈希匹配，三个context全部质量门槛通过，每场84点。证据 `formal/local_parity.json`。第二场raw SHA256 `9ab45cf940e1e7761278daac9a3c1d874d39fff33a50d4fbfbbfd03c51fb8607`。一次本地摘要查询因Python tuple与JSON list过滤不匹配而报空集合，统一JSON表示后重算通过；没有改动采集或分析器。

尾4行同轮B−A竞争增量差（中位数/MAD，us；目标始终A-joint，B是前序）：

| 节点 | 第一场 | 第二场 |
|---|---:|---:|
| M52/W13 h4 | -14.102 / 3.182 | -14.955 / 2.490 |
| M52/W2 h4 | 4.805 / 1.283 | 13.809 / 2.090 |
| M100/W13 h8 | -17.264 / 3.386 | -13.108 / 3.038 |
| M100/W2 h8 | 40.112 / 1.991 | 38.501 / 1.139 |

M100/W2的前序差在两场幅度接近，W13两节点均反方向；M52/W2虽同向但幅度未稳定，不能统一拟合固定历史惩罚。前序A/B/idle的BG-only read中位数第二场为259.246331/262.091998/259.741263 GB/s，write为0.071680/0.076679/0.081169 GB/s；第一场read为262.981852/265.523790/263.367348。两场中B前序背景压力都更高，仍需保留压力混杂。

跨场绝对增量也未稳定。前序A下，M52/W13、M52/W2、M100/W13、M100/W2尾块增量中位数从188.464/95.633/163.667/51.397升至210.119/102.899/180.647/64.222 us，同时对应BG-only读取率下降约3.736 GB/s。前序B下相应增量从174.350/100.121/145.159/91.798升至194.654/116.978/166.519/102.802 us。控制紧前背景不足以消除跨场变化；压力单变量的单调解释也不足以覆盖这些方向。这里尚未扣除冻结响应曲线的预测变化，不能将剩余量命名为已识别缓存成本。

决策：保留需求特征和T0，不新增拟合系数，不采用通用queue修正。下一步从已验收数据分解solo/joint及完整块/尾块的跨场变化，并在冻结曲线支持内量化压力解释项；据此决定需要显式控制的进程/地址状态或连续运行历史。完整计划误差及等预算planner选优验收仍未完成。


### 受控前序跨场漂移的solo/joint拆解（2026-09-15）

直接读取两场已验收raw，选正式31轮PMU-off/timing-on，按context、M、stage、history分别计算solo中位数、joint中位数和同轮joint−solo增量中位数，再做第二场减第一场。跨场不伪配对轮号，不将三个中位数强行作可加分解。`formal/cross_session_decomposition.json`保存全部84节点、两份raw哈希；逐点增量与已验收analysis一致至1e-9 us。

| 分组（跨全部context/M节点） | solo跨场变化范围us | joint跨场变化范围us | 竞争增量跨场变化范围us |
|---|---:|---:|---:|
| W13完整块 | -1.170至2.420 | 10.070至15.430 | 9.010至14.680 |
| W13尾4行 | 0.358至6.874 | 20.130至23.676 | 16.980至23.511 |
| W2完整块 | 0.954至2.889 | 2.235至10.667 | 1.304至9.106 |
| W2尾4行 | -1.862至-0.903 | 6.299至15.201 | 7.266至16.857 |

全部84节点的竞争增量跨场变化均为正，漂移不局限于尾块。完整块solo漂移较小，W2尾块solo还略降而joint上升，因此单纯调高T0不能解释。W13尾块存在局部solo例外：M100在前序A/idle下分别增加6.874/5.991 us，而前序B仅0.700 us，需保留这种基线状态依赖，不能声称所有无竞争点完全稳定。此拆解支持优先调查联合服务状态及其跨场变化，但没有识别出地址、缓存映射、频率或其他具体机制；冻结压力解释项仍待量化，不新增拟合参数。


### 冻结压力曲线的解释项与剩余量（2026-09-15）

复用固定配置实验 `evaluate_followup.predict`，确认当前candidate与原candidate逐项相同，224个已有参考预测全部匹配至1e-9 us。以各场各context的BG-only读取率中位数输入冻结linear/queue曲线；全部压力、完整块h0–7和尾4行h4/h8在原支持域内，无外推/拟合。保存 `formal/frozen_pressure_decomposition.json`，包含predictor/candidate/raw哈希、84个跨场节点和112个同场前序对照。残差定义为观测差减冻结预测差；仅是模型条件残差，不是因果历史成本。

第二场减第一场，queue根据读取率下降预测增量下降，但实测全部上升：

| 分组 | queue预测变化范围us | 观测减预测的残差范围us |
|---|---:|---:|
| W13完整块 | -10.374至-7.974 | 17.205至24.188 |
| W13尾4行 | -18.863至-15.184 | 32.220至40.845 |
| W2完整块 | -6.749至-3.444 | 6.169至14.033 |
| W2尾4行 | -8.846至-4.408 | 15.425至25.703 |

同场B−A尾块在扣除queue压力项后的剩余量（第一/第二场，us）：M52/W13 -28.462/-27.850；M100/W13 -29.843/-24.404；M52/W2 -1.929/+7.762；M100/W2 +36.460/+35.222。linear下M100/W2也为+37.487/+35.562，故这一重复残差不依赖选择queue坐标。M52/W2没有稳定剩余量，不能把原始同向差直接归为稳定历史效应。

这排除了“只需把本次BG-only读取率代入冻结曲线就能解释全部偏差”的方案，但不能排除压力指标本身不充分：BG-only服务率不等于joint期间对前台施加的有效竞争。下一步优先核对按角色每调用事件、背景调用率及全局joint读写率，检查变化来自服务速度还是每调用需求/缓存行为；不把全局DDRC错误分摊到角色，也不把L2 refill字节直接当DRAM需求。


### 角色服务率与每调用事件的跨场核对（2026-09-15）

从已验收raw关联同context/round/cell的PMU-on、timing-off窗口，后台总调用率定义为sum(team_calls)/(native window_ns/1e9)，不使用已平均的calls重复乘除team数；每调用事件复用已验收analysis。每场3context×5条件（BG-only及四个joint前台），每条件31点；结果及raw哈希保存在 `formal/role_service_diagnostic.json`。以下为第二场/第一场中位数比变化，非跨场逐轮配对。

全部15条件的全局读取率下降1.081%–1.420%；bg_local总调用率下降1.141%–1.457%，bg_cross下降1.256%–1.622%。对应后台每调用L2 refill变化分别为-0.009%至+0.087%、+0.006%至+0.119%，而前台每调用refill范围-1.019%至+1.000%。因此后台读取吞吐降低同时伴随调用吞吐降低，并无相应幅度的每调用refill下降证据。不能由较低已实现GB/s推断外生需求更少；观测服务率与可用服务能力/运行状态耦合，是压力单变量解释失败的候选原因。此结果不识别DRAM容量变化，也不证明每调用DRAM字节恒定：L2 refill不是DRAM字节。

另有局部访存行为变化：M52/W2联合窗口的全局写率，前序A/B/idle从0.104535/0.099491/0.110062升至0.182887/0.172996/0.184599 GB/s，绝对增量约0.074–0.078 GB/s；FG每调用L2 writeback从19828/17203/18766升至26934/24726/25296。不能只报告约75%的相对写率增幅而忽略其小绝对值，也不能将全局写流量全归给FG。该局部变化与M52/W2前序残差不稳定相容，但没有因果识别。

下一步应区分需求（每调用工作量及无竞争推进率）和已实现服务吞吐，检查冻结动态模型的输入是否发生这种混用；再确定控制实验，不按本批低吞吐直接下调竞争项或重新拟合容量。完整阶段PMU计数不能定位单个尾块，仍需保持这一限制。


### 动态模型输入语义核对（2026-09-15）

只读检查 `dynamic_resource_candidate/simulate.py`，记录源码SHA于 `receiver_precondition_control/formal/dynamic_semantics_audit.json`。模型已显式区分：声明DRAM预算D、DRAM服务窗口L、请求率q=D/L、实际率x=q*v，以及独立read/total容量约束C。固定点queue响应为 `v_i=1/(1+k_stage*sum_peer x_j)`，排除自身expert；容量超限再做比例限制。因此不能把当前问题概括成“模拟器完全混淆了需求和吞吐”。

尚未覆盖的状态更具体：queue系数和容量在一次预测中固定，响应不依赖紧前背景/进程状态。即使需求D和无竞争成本T0相同，实测显示前序及跨场响应仍不同；当前状态变量无法在相同输入下区分这些观测。实际peer吞吐进入响应反馈本身不等于错误，但不能用BG-only吞吐下降单独识别需求下降或服务能力变化。

接口仍不等价：冻结微基准曲线输入BG-only read GB/s，输出块wall-time增量；动态queue输入joint peer read+write KiB/us，输出DRAM服务时钟速度。单位换算不能修正BG-only/joint、read/total或wall-time/service-time语义差异。不能把新表直接替换k，也不能将每个session的实测吞吐当planner输入以取得表面拟合改善。

下一步实验必须围绕可由计划/运行配置确定的状态构造：先保留同进程、同分配下的前序切换效应，再用独立重启/分配重复区分跨进程变动；同时保留背景服务与每调用事件。若采用会话参考能力校准，必须独立于待预测计划且计入成本，不能用该计划的joint实测反推。当前观测尚不能决定需要容量、请求局部性还是服务延迟状态，暂不新增系数。


### 场内时间漂移与跨场跳变的区分（2026-09-15）

在既有两场31轮分析上，固定取前10轮与后10轮，中间11轮对称排除；这是事后描述性诊断，不作为新增验收门槛或显著性检验。全部168节点及输入analysis哈希保存在 `formal/within_session_time_drift.json`。

各context的BG-only读取率后10轮减前10轮仅为：第一场A -0.057、B +0.082、idle -0.013 GB/s；第二场A +0.060、B +0.029、idle +0.099 GB/s，明显小于跨场约3.4–3.7 GB/s变化。M100/W2尾块的竞争增量后减前，前序A为+0.489/+0.451 us，前序B为-0.534/-1.318 us；对应跨场中位数增加12.825/11.004 us。idle则后减前-9.400/-4.658 us，不能将idle视为稳定状态。

这没有证明重启/分配是因果来源，但不支持用场内平滑时间漂移解释主要跨场差别；两场之间还同时改变进程分配、时间和种子顺序。下一批应采用多个短会话、相同固定调度重复，并在每会话内部保留相同进程/分配的重复块，比较场内与跨重启离散；避免仅再跑两个长会话。保留A/B前序，前台M52/M100×W13/W2及完整块/尾块、solo/joint和PMU/计时控制，集中检验已发现的效应；idle暂不用于定义基线状态。重启同时改变地址分配，若无独立分配控制，只能称为进程/分配联合效应。尚未启动新增实验。


### 固定调度重启对照实现（2026-09-15）

E类独立Lab `receiver_restart_control`，保留native、T0、模型及planner默认；来源文件哈希在source_origin.json。4场（标识616441–616444），每场3预热/12正式轮，前后6轮调度完全相同，各场调度也相同；A/B前序、M52/M100×W13/W2、solo/joint、PMU/计时双开关全部保留。72目标/轮，每目标前500ms预处理，总144窗口/轮。记录native PID，校验每场三个独立角色及跨场不复用PID；同进程内不重启。比较224个同进程块差和336个跨场节点差，后者不伪配对。预计约110分钟，硬上限180分钟，在既有≤4h授权内。

本地调度/完整raw校验12项测试通过（2.80s），四场比较及重复PID/会话拒绝4项通过（4.79s）；新增/修改核心Python Ruff通过，runner bash -n通过。待目标机同套检查及真实native smoke；尚无本批性能结论。失败保留原目录，不自动重启或改门槛。回退边界是独立Lab目录，旧受控前序原始数据保持不变。


目标机16项测试通过（17.21s），6份关键采集/校验源码哈希与本地一致，FG/BG native身份与已验收上一批相同。manifest全文件YAML解析因原有第785行未引用冒号文本失败；移除本次新增条目后仍同样失败，新增feature/variant各自解析通过，未修改无关条目。定向diff --check与runner bash -n通过。

依据既有≤4h授权，180分钟流程已由事件通知启动器启动：本地supervisor PID60891，远端timeout PID4059823已一次性确认存活；状态 `receiver_restart_control/notification_run/`，输出 `receiver_restart_control/formal/`。先进行真实native smoke，门槛通过后自动进行四场；目前尚无smoke通过或正式结果。design SHA256 `f7968c28fd3d6a31aeb9d443bf768d4d89e36ec93a26911adfedbac65ce4edf1`，runner `68464ac42e0e7f2662d74c9a56fb04863711a5880f57b4544423c2737c0600bc`。完成通知负责续接，不重复启动或投递。


固定调度重启批次真实native smoke完成并本地独立验收：全部冻结源码哈希匹配，analyze_control从smoke raw重算与远端JSON完全一致，A/B两context全部质量门槛通过。证据 `receiver_restart_control/formal/smoke.local_parity.json`，raw SHA256 `0c2e06ca50f57f724296923c5f5ed8f4ccd2217ea9dfaec9400506d96577940c`。远端日志随后进入第一场预热（round -2/12）。这只验收采集集成，四场重复性及模型验收尚未完成。


### 固定调度重启对照第一场验收（2026-09-15）

第一场12轮及完整终态已下载，本地analyze_control重算与远端JSON完全一致，A/B所有质量门槛通过。raw SHA256 `8e3c1344f5499e25e0b47b100f38003221e48ad6727ad6156d688e7f474e431b`，native PID4066016/4066017/4066018；证据 `receiver_restart_control/formal/session1.local_parity.json`。原远端timeout PID4059823在elapsed30:19确认存活；此时日志首场12/12及all_gates_passed True，未重启采集。

相同调度位置的后6轮减前6轮，竞争增量配对变化中位数：W13完整块范围-4.785至+3.935 us，W13尾4行-4.399至+0.809；W2完整块-1.056至+1.770，W2尾4行-1.571至+0.578。M100/W2尾块前序A为-1.358 us（MAD0.824），B为-1.571（MAD1.153）；仅6对，不作显著性结论。该场未出现此前跨场约11–13us的变化，不能据此宣称跨进程稳定；其余三场及最终四场对照仍待验收。保留配对差中位数与各块中位数，两者不能直接互换。


### 固定调度重启对照第二场独立验收（2026-09-15）

原timeout PID4059823在elapsed57:57仍存活，第二场12/12及分析已关闭后下载对应raw和analysis；没有重启或复制未关闭raw。本地冻结analyze_control独立重算，JSON规范化后与远端完全一致，A/B质量门槛全部通过。raw SHA256 `ee6fe3bc6d3caddd516853945060c9f3574f81706defbab819520152cec61ef4`；native PID4156595/4156596/4156597，与首场共6个PID互不重复。证据 `receiver_restart_control/formal/session2.local_parity.json`，包含56个场内配对差和56个第二场减第一场的非配对中位数差。一次终端摘要误用字符串stage过滤数值stage导致空集合；改按0=W13、1=W2读取后摘要成功，原始验收及产物未受影响。

| 节点组 | 第二场场内后6轮减前6轮，配对差中位数范围us | 第二场减第一场，非配对增量中位数差范围us |
|---|---:|---:|
| W13完整块 | -2.870至2.120 | -2.655至14.880 |
| W13尾4行 | -3.279至-0.377 | 4.284至19.624 |
| W2完整块 | -0.663至0.965 | -0.837至3.090 |
| W2尾4行 | -1.498至0.773 | -5.019至3.459 |

B前序M100/W13尾块增量本场中位数185.284 us，比首场高19.624 us，本场内部配对变化-3.279 us；A前序相同节点跨场增加6.508 us。固定调度后依然观察到跨场差异，且并非所有stage/context统一上升。该初步结果不能归因为重启、地址或硬件状态：时间与进程/分配仍混杂，压力及角色服务指标尚未完成四场联合分析。剩余两场与最终四场比较尚待验收，不修改T0或拟合系数，不据此声称模型或planner改善。


### 固定调度前两场服务指标补查（2026-09-15）

对两场已验收analysis的PMU-on/timing-off记录，分别提取A/B前序下BG-only及四个joint前台条件；每条件验证12个不同轮号，保留analysis SHA和中位数/MAD。证据 `receiver_restart_control/formal/session1_2_service_diagnostic.json`。不使用块计时窗口冒充同步PMU观测，不新增拟合或验收门槛。

BG-only read GB/s：前序A第一/二场259.276690/259.183815（MAD0.126825/0.045314），前序B262.187515/262.042281（MAD0.107180/0.178124）。跨场仅下降0.036%/0.055%，与上一批受控前序约1.1%–1.4%的全局读取率变化不同。两场全部10个BG-only/joint条件的read变化为-0.236%至-0.011%；bg_local/bg_cross每调用L2 refill变化合计范围-0.059%至+0.087%，前台为-0.515%至+1.086%。

因此本批W13尾块跨场+4.284至+19.624 us发生时，整窗口BG-only服务率变化很小；不能简单沿用上一批“较明显的全局服务率漂移”解释。B前序M100/W13的FG每调用refill增加1.086%，但这是完整stage计数，无法定位尾块，也不能把L2 refill等价为DRAM需求。前两场证据仍无法识别缓存映射、地址、请求时序或具体服务延迟机制；须保留局部状态/服务时序尚未观测的可能性，等待四场对照后决定下一项控制实验。不能据此宣布等压力成立或引入session标签拟合。


### 固定调度前两场按轮块增量求和（2026-09-15）

为判断尾块异常在较大计算范围内的重要性，对已验收analysis按context/M/stage收集全部history，验证M52为4个完整12行块+尾4行、M100为8个完整块+尾4行，每节点12轮。先在同一轮内求和joint−solo块增量，再取中位数；没有把各块中位数相加。证据 `receiver_restart_control/formal/session1_2_block_sum_diagnostic.json`，含来源analysis哈希、逐轮完整块/尾块/总和。该量是已记录块竞争增量之和，不是独立测量的完整GEMM wall time，不包含未计时开销；百分比的分母是竞争增量，不是总计算时间。

| 前序/前台 | 首场块增量总和中位数us | 第二场us | 跨场变化us |
|---|---:|---:|---:|
| A/M52/W13 | 590.830 | 608.854 | 18.024 |
| A/M52/W2 | 291.488 | 300.731 | 9.243 |
| A/M100/W13 | 822.928 | 842.606 | 19.678 |
| A/M100/W2 | 397.221 | 401.274 | 4.053 |
| B/M52/W13 | 591.105 | 625.667 | 34.562 |
| B/M52/W2 | 308.357 | 310.739 | 2.381 |
| B/M100/W13 | 906.640 | 997.542 | 90.902 |
| B/M100/W2 | 509.995 | 514.873 | 4.879 |

B/M100/W13的完整块增量之和中位数跨场增加70.525 us，尾块增加19.624 us，总和增加90.902 us；三者中位数不强制可加。第二场总和内部后6轮减前6轮配对变化仅+0.072 us。该节点问题不仅在尾块，完整块累计变化更大；只修尾块会遗漏主要累计偏差。90.902 us约为首场竞争增量总和的10.03%，不能称为完整GEMM或计划10%的误差。四场及真实trace验证未完成，保持诊断结论，不新增系数。


### 固定调度W13完整块跨场差的history位置（2026-09-15）

直接查询两场已验收 `session*.analysis.json` 的逐节点delta_median_us，按context/M/stage/history匹配；以下为第二场减第一场，单位us，不是将history当独立重复样本作统计检验。

| 前序/前台 | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | 尾4行 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A/M52/W13 | 2.460 | 2.120 | 2.425 | 3.590 | — | — | — | — | 5.810 |
| B/M52/W13 | 2.970 | 4.650 | 9.455 | 14.880 | — | — | — | — | 4.284 |
| A/M100/W13 | 1.440 | 2.845 | 5.305 | -0.450 | 0.235 | -2.655 | 6.500 | 1.790 | 6.508 |
| B/M100/W13 | 2.835 | 5.890 | 10.685 | 9.990 | 9.990 | 8.615 | 11.800 | 10.460 | 19.624 |

B前序的差异并非集中在首块：M100后续h2–7完整块增加8.615–11.800 us，明显高于h0的2.835；M52也在h2/h3更大。A前序没有相同幅度及一致形态。因此后续模型诊断应保留前序×接收方history的交互，不能先把跨场差写成统一首块冷启动偏置或只修尾块。这里history只代表块访问位置，不证明这些位置已处于固定的冷/热缓存状态；仅两场，未证明可泛化交互，更不能将session标签作为planner特征。四场重复性与连续真实trace迁移仍待验收。


### 固定调度第三场独立验收及回落（2026-09-15）

原timeout PID4059823在elapsed1:25:06存活，第三场12/12及all_gates_passed True后下载已关闭raw/analysis。本地冻结analyze_control从raw重算，JSON规范化后与远端完全一致，A/B全部质量门槛通过。raw SHA256 `60b050ee8f80adce53ba52e36f6d7b5496bf72baa3fc1663b3f67444327a3d3a`，native PID55580/55581/55582；三场共9个PID不重复。证据 `receiver_restart_control/formal/session3.local_parity.json`。第四场及四场终态尚未验收，不能将前三场等同完整批次。

逐轮先求块竞争增量之和再取中位数，沿用此前口径（不是独立whole-GEMM wall time），三场分别如下：

| 前序/前台 | 第一场us | 第二场us | 第三场us |
|---|---:|---:|---:|
| A/M52/W13 | 590.830 | 608.854 | 592.032 |
| A/M52/W2 | 291.488 | 300.731 | 291.769 |
| A/M100/W13 | 822.928 | 842.607 | 836.307 |
| A/M100/W2 | 397.221 | 401.274 | 401.296 |
| B/M52/W13 | 591.104 | 625.667 | 593.661 |
| B/M52/W2 | 308.357 | 310.738 | 309.243 |
| B/M100/W13 | 906.640 | 997.542 | 922.500 |
| B/M100/W2 | 509.995 | 514.873 | 515.729 |

B/M100/W13尾块增量三场165.659/185.284/165.752 us；第三场内部总和后6轮减前6轮配对变化-2.633 us。第二场的明显抬升没有在第三场保持，不能直接变成稳定history修正或按会话序号单调增加的惩罚。第三场BG-only read A/B为259.401706/262.122784 GB/s，仍接近前两场。当前证据支持进一步处理会话间响应差异，未定位到进程/分配、时间或硬件机制；PID回绕不构成重启证据。三场逐轮总和、尾块和服务率及输入哈希保存在 `formal/session1_2_3_block_sum_diagnostic.json`。模型、T0及planner保持冻结，等待第四场独立验收和预定义完整比较。


### 固定调度四场全部独立验收（2026-09-15）

远端原timeout进程已退出，formal/COMPLETE和comparison.json存在，日志四场全部all_gates_passed True，最终224个场内/336个跨场对照完成。下载最终产物后，本地冻结analyze_control重算smoke及四场raw，再由compare_restart.compare重算四场比较，JSON规范化后全部与远端完全一致；source.sha256全部匹配、12个native PID互不重复、冻结门槛全部通过。证据 `receiver_restart_control/formal/local_parity.json`。第四场raw SHA256 `54797fe362af14e4935a823004e9e72754ca8ca5d153affe6cc84d86749a6b5f`。

注意通知链路与远端结果分开：本地supervisor60891和SSH60892在elapsed1:55:30仍存活，completion.json/notification.json尚无；不能声称通知已成功或SSH已返回code0。远端已完成产物足以开展独立验收，不重跑采集、不重复投递。需后续只读定位为何SSH通道尚未关闭。

四场诊断保存在 `formal/four_session_diagnostic.json`，按轮求全部块joint−solo增量之和再取中位数，仍非独立whole-GEMM wall time：

| 前序/前台 | 第一场us | 第二场us | 第三场us | 第四场us |
|---|---:|---:|---:|---:|
| A/M52/W13 | 590.830 | 608.854 | 592.032 | 610.091 |
| A/M52/W2 | 291.488 | 300.731 | 291.769 | 307.726 |
| A/M100/W13 | 822.928 | 842.607 | 836.307 | 878.207 |
| A/M100/W2 | 397.221 | 401.274 | 401.296 | 408.120 |
| B/M52/W13 | 591.104 | 625.667 | 593.661 | 624.029 |
| B/M52/W2 | 308.357 | 310.738 | 309.243 | 313.726 |
| B/M100/W13 | 906.640 | 997.542 | 922.500 | 1004.158 |
| B/M100/W2 | 509.995 | 514.873 | 515.729 | 529.048 |

B/M100/W13跨场范围97.518 us，各场内部配对总和变化+6.132/+0.072/-2.633/+1.637 us。BG-only read A四场259.276690/259.183815/259.401706/259.311712 GB/s，B为262.187515/262.042281/262.122784/262.154432；整体服务率接近仍不足以区分较大响应差。四场不能证明双峰、奇偶会话效应、地址或硬件机制，时间/进程/分配继续混杂。

尾4行同轮B−A增量差中位数四场：M52/W13 -13.059/-15.484/-15.301/-14.280 us；M52/W2 +5.546/-0.072/+6.298/-1.819；M100/W13 -13.876/-2.114/-14.299/-4.360；M100/W2 +40.880/+45.249/+42.957/+47.304。M52/W13与M100/W2方向及幅度较可重复，其他节点不能统一处理。前序B仍改变BG-only服务率，以上不是已辨识的等压力缓存成本。

模型决策：保留T0和需求特征，不用session标签、奇偶标签或整批平均新系数掩盖变化，不将所有尾块归入统一惩罚。后续应优先在同一进程控制权重/工作区分配变化，或使用独立于待预测目标的参考测量检验可校准状态；先辨识对已有90us量级完整块累计差的解释力，再进行连续真实trace迁移及完整计划/等预算planner门槛。此次仅完成重复性诊断，不等于新模型验收。


### 四场完成后的通知链路与分配控制可行性审计（2026-09-15）

只读检查远端进程表：原sshd会话4059821（父4059806）仍在，无直属子进程，也无本批foreground_native/background_native/receiver_restart_control采集器残留。本地60892仍等待，子进程60893为经jtmeng-jumper-internal的ssh -W代理；因此客户端无直接TCP结果不代表已退出。读取远端/proc/4059821/fd被权限拒绝，未扩大权限或据此认定根因。通知尚无completion/notification记录，暂定位为命令产物已完成而SSH会话未关闭，不能声称已查明代理故障。不重启采集、不伪造code0、不重复投递。

读取冻结native源码确认：FG和BG的a13/a2/b13/b2/c13/c2在命令循环之外分配并复用；每个窗口在循环内重置输出、创建TeamState/records及worker线程，再join。故四场设计中“同进程/同分配”只适用于主要矩阵缓冲区，不代表线程或所有临时分配保持相同。现有窗口命令没有更换矩阵分配的控制项，分配实验需要独立Lab native变体，不能仅改Python标签声称地址已改变。

下一项实验约束：先仅控制FG主缓冲区，BG保持原进程/原分配/原协议；在进程启动时准备等形状、等内容、等对齐的多个FG缓冲区集合，记录集合身份及地址/尺寸（虚拟地址不能冒充物理映射），测量窗口之外选择集合。用均衡的集合切换与原集合返回检查可逆性，每个集合内均保留A/B前序及匹配solo/joint；M100/W13为主要诊断，M100/W2为前序效应对照。保留原PMU/timing门槛和最小native数值/CPU正确性检查。额外驻留内存本身是混杂，必须包含同一新进程中固定集合不切换的对照；既有native vs新native固定集合先验证观测扰动。尚未实现或启动此批，不新增模型参数。若仅FG切换不能解释变化，再独立控制BG，避免同时改变所有缓冲区而无法定位。


### 前台分配控制native实现与目标机正确性（2026-09-15）

E类独立Lab `receiver_allocation_control` 已新增foreground.cpp、buffer.h、build.sh、test_native.py、design.json及source_origin.json。复制的是小型Lab探针，旧native及生产内核不变。回退边界仅新目录和对应manifest/model诊断说明。应用impact-analysis/test-selector/code-review-gate；CodeGraph对tmp目标无有效索引后使用定向读取。只新增实验命令BANK_COUNT=1/2及窗口第8项bank，不更改公共接口或生产默认。

启动时分配1或2套A13/A2/B13/B2/C13/C2，使用4096字节对齐、同内容初始化，进程存活期间保留，窗口外绑定所选集合引用；结果输出bank_count/bank及六个地址/字节数。地址仅代表虚拟身份；对齐方式与旧vector不同，额外常驻内存及物理分配都是待控制因素。原计时区间从begin到iteration更新前的代码去除空白后与旧探针一致；未改内核/块计时/同步结构。原每窗口worker重建仍保留。

Arm-codex-internal上以NUMA3构建，原GCC C++17/O3/SVE256/BF16选项不变；构建绑定240–243，测试子进程绑定240/319、membind3。命令 `timeout 5m numactl --physcpubind=240-243 --membind=3 bash tmp/joint_cost_model_20260911/receiver_allocation_control/build.sh`，随后 `timeout 5m .venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_allocation_control/test_native.py`：8 passed in42.33s。有效112窗口覆盖M16/48/52/76/96/100/112、W13/W2、timing0/1、bank0→1→0，验证地址稳定、4096对齐、集合间非重叠及原数值/guard/CPU检查；6负例覆盖非法集合数量/编号及截断窗口命令。不是性能测量，不用20ms正确性窗口替代正式500ms门槛。

native SHA256 `37cadb7fa3b617790d407d5db47f2e5bdf6d7a9908e9cc36d2a0fe4e326382e5`，foreground源码SHA256 `51b4c8e350e6a558964964b4163b700ddf961896a0e00d160fc789c19746307f`本地与远端一致；构建身份与结果在build_identity.txt/native_validation.json。Ruff检查、格式化、bash -n及定向diff --check通过。本地macOS的8项native测试按目标约束跳过，不称为本地通过。manifest原有YAML错误未修复，新条目仅登记native阶段，正式benchmark尚未就绪。

尚未完成：带bank身份及完整性校验的配对采集器、固定集合的新旧探针扰动对照、正式A/B前序×集合切换采集。后续先完成这三步，保持原质量门槛和已有≤4h授权边界；不能根据本次数值通过声称分配解释了跨场误差或planner改善。


### 分配控制配对采集器接入（2026-09-15）

新Lab复制并保留上一批原始质量分析器依赖及来源哈希，新增allocation.py身份校验、三FG actor采集、八context分离分析和compare_allocation.py。三个FG actor为旧探针、新1-bank、新2-bank，后台local/cross两个进程始终共用；全部actor常驻，任一目标只激活选定FG。context为A/B前序×old/single/dual0/dual1，FG聚焦M100/W13和M100/W2，完整timing×PMU开关、BG-only/idle目标控制保留。每轮160目标，每目标独立500ms前序，共320窗口；solo/joint相邻并按轮平衡顺序，6轮调度重复，12轮正式/3预热，两种子616451/616452，smoke616450为1/0。

raw记录5个唯一进程PID和三个actor的binary/argv身份；每个新FG窗口验证bank_count/bank、六个buffer地址及精确尺寸、4096对齐、同进程稳定与非重叠。不同进程虚拟地址可相同，不跨进程错误判重。按context先检查原始协议，再显式构造该actor的预期协议传入冻结质量分析，保留原数值、CPU、计数分辨率、PMU和timing门槛。没有将bank标签当模型输入。

本地完整raw/错序/缺失/bank/地址漂移与重叠/进程/协议/数值/PMU等18项测试通过（7.70s），远端同18项通过（17.03s）；比较器4项通过（4.12s），包括非零bank响应差方向、重复PID/会话拒绝。预定义216个同轮variant差、288个同进程重复差和144个跨场非配对差。Ruff检查及runner bash -n通过。

真实native smoke已通过交互工具会话7814启动，输出 `receiver_allocation_control/smoke/raw.jsonl`，8分钟硬上限，NUMA3/CPU240–319、500ms窗口；当前尚未得到终态或smoke门槛结论。正式180分钟fail-fast runner已准备，尚未启动；启动前必须独立验收smoke并验证冻结源码。多个FG的额外驻留内存及旧/新allocator差异仍保留为实验限制，不能以同后台就声称纯物理映射因果识别。


### 分配控制smoke独立验收并启动正式两场（2026-09-15）

真实smoke交互会话7814正常code0，round1/1和all_gates_passed True。下载关闭的raw/analysis/source清单，本地analyze_control独立重算与远端JSON完全一致，所有冻结源码哈希匹配；8个context/144响应节点、single:0/dual:0/dual:1身份检查全部通过。最大PMU扰动0.600783%，块计时扰动0.541105%，单调用计数占比2.272727%，原门槛均通过。证据 `receiver_allocation_control/smoke/local_parity.json`，raw SHA256 `4f890053ef99f21d1922b91b3edb9ef89c6308557b88392f762e7b83a63961b7`。smoke仅证明采集集成，未证明跨轮重复性或模型泛化。

目标机比较器4项测试通过（9.86s），runner bash -n通过；runner SHA256 `6aee5c9ac51e08e389b2e4a8e2616c55d9a37479da37e424bbcc14a906d8ce9d`，design `8162618c07ee69a6252ac04c211702255de787a82d281c44eadff940e3d04333`本地/远端一致，formal输出目录预先确认不存在。审查新代码保持原质量门槛、完整raw再分析、输出独占、进程清理及native身份检查；正式数据仍缺失，无模型验收结论。

依据既有≤4h授权，由run_experiment_notify.py启动180分钟硬上限正式流程，预计约130–160分钟（smoke约5分钟/轮×两场各15轮，含启动/分析余量）；本地supervisor70501，状态 `receiver_allocation_control/notification_run/`。新SSH启用ServerAliveInterval15/ServerAliveCountMax3，帮助识别断连，但不承诺解决所有通道退出问题。两场完成后自动重算并生成216个variant、288个场内和144个跨场对照，再写COMPLETE；任一步失败不自动重启、不放宽门槛。

旧receiver_restart_control四场已经独立验收。仅在确认原远端无计算子进程后，关闭残留本地SSH60892并记录transport_cleanup.json；原通知程序实际记录SSH255/remote_state_unknown，于22:30:02 UTC完成一次queue投递且accepted。该状态是通道返回值，不推翻此前COMPLETE/raw独立验收；不伪造code0、不重跑或重复投递旧批次。


### 分配对照的结果判读边界补充（2026-09-15）

正式批次仍待完成，本节仅核对已冻结设计和分析口径，不提前报告性能结论。当前三个actor是old、single（bank_count=1）及dual（bank_count=2，bank0/1交替参与调度）。此前“同一新进程中固定集合不切换”的要求未被完整实现：single覆盖新探针固定单集合，但没有保持双集合驻留且始终只使用一个bank的独立对照。因此single与dual同时改变了集合数量、进程/分配身份及访问历史；dual0与dual1则共享切换历史。当前批次可以测量这些条件下的关联，不能单独辨识切换成本、额外驻留内存成本或物理地址映射原因。保留正在运行的冻结批次，不在运行中补标签、改协议或放宽门槛。

验收后按每场、前序A/B、W13/W2、完整块history及尾4行分别保留solo、joint与joint−solo，版本差均按同轮配对。先逐轮求差或求块总和，再计算median/MAD；不可用独立中位数的相减代替配对差，也不可把块总和冒充独立whole-call时间。需分别检查两个正式会话的方向/幅度，以及各场后6轮相对前6轮的变化，不能只报告两场混合后的平均效果。两场会话不足以估计一般跨进程分布。

原始数据重建检查已在本地smoke/raw.jsonl（上述SHA256，1轮）完成：A/B×W13/W2×3种版本对照×9块共108项，逐项满足“joint变化=solo变化+竞争增量变化”，最大数值残差0 us；对应窗口的块总和加overhead与mean_ns记账检查通过。这里只验证分析可执行性，不将单轮smoke用作效应或泛化证据。

后续决策按证据分支：若版本差主要体现为solo变化，优先检查探针/无竞争基线；若joint−solo仍有跨会话可重复的变化，保留为竞争响应状态候选，并在引入模型前补足同驻留量固定bank与切换的对照；若方向不稳定，则保留为未解释变异，不以bank/session标签拟合。即使本批关联成立，也必须经过连续真实trace、完整计划预测及等搜索预算planner验证，才能支持总体模型改善。


### 四会话M100/W13整次调用口径交叉检查（2026-09-15）

只读重查receiver_restart_control/formal/session1–4.jsonl，四个raw SHA256逐一匹配local_parity.json。选择1T/M100/W13、B前序、block_timing=0、PMU关闭的mean_ns（完整调用均值），每场12个正式轮次；同轮配对joint−solo后取median/MAD。此口径独立于此前开启块计时的块总和，仍是窗口内重复调用均值，不是连续expert trace。

| 会话 | solo中位数us | joint中位数us | 配对竞争增量中位数us | 增量MAD us |
|---|---:|---:|---:|---:|
| 1 | 10291.900 | 11205.650 | 912.800 | 3.900 |
| 2 | 10283.600 | 11286.050 | 1002.150 | 4.050 |
| 3 | 10286.750 | 11210.250 | 925.550 | 5.100 |
| 4 | 10283.800 | 11290.100 | 1005.500 | 8.250 |

无块计时条件下，竞争增量跨场范围92.700 us，solo范围8.300 us，joint范围84.450 us；支持跨场差异主要在联合响应，不能仅通过移动无竞争基线消除。独立中位数不可强行相加，范围也不要求可加。竞争增量的变化约为该增量的10.2%，但相对约11.2 ms的整次joint调用仅约0.83%；这一级别诊断不能被表述成整个计划存在同等比例误差，更不能代替此前约11–15%的宽/窄计划低估定位。最终是否值得增加状态复杂度仍以真实trace与完整计划/选择误差改善为准。没有据此修改T0、模型参数或验收阈值。


### 完整计划1T误差按M分组复核与后续优先级（2026-09-15）

在分配实验等待期间，只读重分组既有receiver_mixed_holdout/phase_diagnostic.json（SHA256 `e1d8560a6d930c7f6482908eef6da6b1205105852a10aa471309623b8570352e`）。这是对已测混合宽度留出的事后诊断，不是新留出、不修改参数。每expert样本为31次阶段包络的均值；复核逐项predicted_us−actual_us=error_us，按mixed_narrow/1T/M区间分组，分别报告两场。下表为每expert绝对误差的算术平均，正数高估、负数低估；这些不是lane关键路径贡献，不能加总成makespan误差。

| M范围 | 每场expert数 | W13平均误差us，场1/场2 | W2平均误差us，场1/场2 |
|---|---:|---:|---:|
| 1–11 | 31 | -12.28 / +4.51 | +272.81 / +274.28 |
| 12–24 | 15 | -1557.26 / -1596.09 | -224.86 / -223.14 |
| 25–48 | 11 | -2636.73 / -2619.68 | -1243.16 / -1303.36 |
| 49–96 | 12 | -1935.92 / -1914.59 | -1216.89 / -1222.92 |
| ≥97 | 13 | -2009.80 / -2015.46 | -1535.30 / -1446.14 |

M25–48逐expert相对误差均值：W13 -42.04%/-41.88%，W2 -39.27%/-40.33%；M12–24 W13 -42.25%/-42.65%。M1–11 W2则+77.18%/+78.97%。M1–11 W13绝对平均接近0但逐expert相对均值+17.59%/+19.63%，说明存在量级/方向抵消，不能将该组宣布准确。各expert的M、team位置、访问历史与动态重叠并未独立控制，分组差异不能直接归因为kernel或冷/热B机制。

这一量级复核改变下一步诊断优先级：当前M100分配批次保留并按既定规则验收，但其会话波动并不足以替代主要残差定位。验收后优先针对真实计划中1T/M12–48的接收响应、完整块/尾块以及竞争阶段时序，与已测同route隔离基线匹配；同时保留M1–11 W2高估作反向约束。不可统一提高1T惩罚，也不可把这些已见真实联合时间直接回填为“可泛化”成本。新修正必须在独立条件拟合后再做未见动态重叠/完整计划与等预算选择验证。


### M16/1T真实与预测阶段重叠核对（2026-09-15）

只读receiver_mixed_holdout/jobs.json、predictions/mixed_narrow.jsonl及session1/2_compact.json，目标expert203/M16/1T/core_begin6。预测31arrival与两场各31实测分别汇总，不把arrival_index与实测pair强行配对。对目标自身的阶段区间，累计每个其他expert对应阶段与之相交的时长，再除以目标阶段时长，得到时间加权平均并发team数；按core_begin//40区分同LLC和跨LLC，排除目标自身。这是阶段活动重叠，不是请求率或实测DRAM压力；gather使用阶段包络，内部worker空隙未扣除。

| 目标阶段/来源 | 开始时间均值us | 时长均值us | 同LLC W13 team均值 | 同LLC W2 team均值 | 跨LLC W13 team均值 | 跨LLC W2 team均值 |
|---|---:|---:|---:|---:|---:|---:|
| W13/预测 | 21174.973 | 1839.115 | 9.596 | 9.005 | 9.848 | 9.473 |
| W13/实测场1 | 25320.344 | 3971.645 | 11.819 | 6.680 | 9.546 | 9.930 |
| W13/实测场2 | 25174.147 | 4015.419 | 11.859 | 6.611 | 9.607 | 9.878 |
| W2/预测 | 23014.477 | 1068.305 | 9.700 | 8.508 | 13.235 | 6.470 |
| W2/实测场1 | 29293.401 | 1868.645 | 7.945 | 8.822 | 12.615 | 6.780 |
| W2/实测场2 | 29191.016 | 1915.259 | 7.903 | 8.890 | 12.480 | 6.938 |

目标W13期间平均其他GEMM team总数预测37.922，实测约37.975/37.955，数量几乎相同，而同LLC的W13/W2构成变化明显；W2期间预测总数37.913，实测约36.162/36.211，并非实测有更多并发team即可解释低估。该结果只否定“漏掉大量并发任务数”这一简单解释，不否定相同team数下M/width/history、发出强度及接收敏感度的差异。预测与实测目标开始时刻已有约4–6 ms差异，因果方向仍混杂，必须同时保留自主时间线误差。

原始trace为阶段事件，预测额外含请求segment_events，但当前实测记录未提供对应块请求时间戳。后续固定实测阶段边界只能作为条件诊断；将独立请求预算摊入阶段仍是建模假设，不能命名为实测请求时间线。当前没有完成该条件响应回放，也没有据本次重叠统计拟合参数。


### 固定预算、阶段均匀发出假设的压力对照（2026-09-15）

继续使用上一节相同jobs、31arrival预测和两场各31实测阶段边界，固定每expert/stage已有read+write KiB预算，将预算均匀摊入各自阶段包络，按与expert203目标阶段的重叠积分并排除目标自身。这是显式均匀发出假设下的条件统计，不是原模拟器segment级有效压力、不是真实DRAM观测，也不是自主预测。实测时长进入预算/时长分母，已包含减速结果，存在内生性。

目标W13平均peer压力：预测阶段重建286.807 KiB/us，实测两场阶段重建187.435/185.053；W2分别294.359、208.472/203.697。同LLC W13分别136.186、94.230/93.296；W2分别139.044、100.546/99.780。在该假设下，替换为实测阶段并不会产生更大的平均压力；实际时长更长会将固定预算摊薄。因此不能把真实执行时间拉长后得到的较低请求率，用来证明硬件竞争更弱或直接反拟合更大的接收系数。

代入冻结queue项的1+k*平均peer压力，仅得到W13约1.00717（预测重建）及1.00469/1.00463（实测重建），W2约1.01472及1.01042/1.01018。这只是queue项在平均特征处的乘数，不包含容量缩放、segment瞬时变化或时钟重叠，不能当作完整模型slowdown。结果提示：阶段边界本身不足以恢复请求服务压力；下一步必须保留独立请求预算/时间分布及接收敏感度的辨识，不能把“固定实测阶段”当作已完成对时间线与响应的精确归因。未改模型、参数或冻结采集。


### 当前受控背景与真实窄计划的组成差异（2026-09-15）

沿用expert203/M16/1T的31次阶段重叠统计，再按竞争者width分组（仅W13/W2活动包络，排除gather与目标自身）。目标W13期间，同LLC平均活动1T/2T/4T team为8.967/4.727/4.805（场1），8.966/4.731/4.774（场2）；跨LLC为9.792/4.830/4.854和9.798/4.829/4.857。按width加权的GEMM活动核数约37.641/38.868（local/cross场1），37.524/38.884（场2），是阶段活动统计，不能当作100%访存占用。

当前receiver_allocation_control的目标背景是每域6个4T team，即24核/域、12个同构team总计；A/B只控制此前后台前序。它没有复现真实窄计划接近满域的1T/2T/4T混合活动构成，也没有复现动态W13/W2混合。不能从本批分配对照外推真实M16约2.3ms的联合增量已经被辨识。此限制针对当前控制批次，不否认其他历史实验已覆盖的条件。

待本批验收后，主要残差诊断应增加与真实计划拓扑一致的混合背景：目标所在域保留其1核，其他19个team按9×1T、5×2T、5×4T（39核）；另一域10×1T、5×2T、5×4T（40核）。这仅是来自固定计划布局的候选拓扑，尚未指定各竞争者M/阶段/时间序列、未实现或启动，也不把时间加权平均数量直接取整当作真实trace。后续必须从独立预算与已记录的形状/阶段分布定义可复现协议，保留同条件solo和小M反向对照，不以宽度惩罚代替竞争机制。


### M16/1T冻结请求分段来源核对（2026-09-15）

读取receiver_mixed_holdout/jobs.json中mixed_narrow/expert203。W13成本分为M12前缀1210.58 us与M12→16增量469.75 us；W2为622.95与218.73 us，分别守恒到1680.33/841.68 us。请求也有两个segment，但M16累计请求节点明确为unvalidated_interpolation，source_m=[12,48]；M12为measured。W13读预算分别7777.120与1236.230 KiB，W2为3816.927与586.957 KiB。第二段是累计请求曲线的差分，不是直接测得的“同次调用第2块/4行尾块”请求预算。

physical_gemm_provenance同时标记window_fraction=1.0、window_status=explicit_unvalidated_assumption，即把每段全成本时长作为请求服务窗口。故目前该点包含三个应分开验证的因素：已由同route隔离对照约束的总成本、未经直接验证的M16请求量/段内分布，以及不随M/history变化的DRAM接收系数。不能仅凭隔离时间准确就宣布请求特征准确；也不能先认定缺陷全在响应系数。

旧sensitivity字段第二段为0，但separate DRAM queue路径覆盖旧响应公式，不使用这个字段；不能误诊为“尾块sensitivity=0导致完全不计竞争”，尾块仍有DRAM预算和服务时钟。后续对M16及M25–48的诊断需分别核验累计预算与同次调用块分布，再接入混合背景响应。没有替换插值、修改成本或启动测量。

上述M分组、M16阶段重叠/width构成及均匀请求条件统计已保存为 `receiver_mixed_holdout/m16_exposure_diagnostic_20260915.json`，SHA256 `ecd7309c5336d15804149ee3e85b5471396286acdca41be83ec4bc1732c3496a`。产物保留93条样本时间线诊断、20个M分组、目标请求来源和10个输入文件哈希；两场compact所列原始trace/session/frontier哈希逐一核对，样本身份唯一、按width与按stage累计的并发量一致、误差恒等式及JSON回读检查通过。仅为既有数据的诊断产物，不改变冻结预测或模型。


### 1T请求节点覆盖审计与下一轮辨识顺序（2026-09-15）

只读冻结receiver_mixed_holdout/jobs.json，按最终累计M节点的request_node_modes审计两种混合计划；不是每个前缀节点都未测，也不是新增性能测量。W13/W2的分类计数一致。mixed_narrow共82个1T expert：measured 3（均M12）、unvalidated_m12_small_m_anchor 31、unvalidated_interpolation 35、unvalidated_extrapolation 13。mixed_wide共20个1T expert：measured 0、小M锚点8、插值8、外推4。该计数仅说明请求特征验证覆盖，不能把“未验证”直接等同于“错误”。

窄计划M1–12使用相同累计read+write预算：W13 7777.984 KiB、W2 3817.322 KiB。Q/T0的名义独立需求率如下；它不是实测联合带宽，也没有包含模拟器的容量缩放及动态服务时钟。

| M | W13 Q/T0 KiB/us | W2 Q/T0 KiB/us |
|---|---:|---:|
| 1 | 18.003 | 18.887 |
| 4 | 15.962 | 16.177 |
| 8 | 10.165 | 9.797 |
| 11 | 6.428 | 6.078 |
| 12 | 6.425 | 6.128 |
| 16 | 5.365 | 5.239 |
| 35 | 4.104 | 3.899 |

这揭示了一个必须与接收敏感度分开的假设：小M以更短T0发出同一请求预算，可能改变容量竞争的计算结果。但packed B大小本就不随M线性缩放，因此不能根据M小就认定预算应按M/12缩小，也不能据此宣布小M/W2高估原因已解决。M16/35的请求预算分别依赖M12–48插值，与同次调用完整块/尾块的实际请求分布仍有区别。

下一轮先核验独立需求，再拟合响应：建议训练诊断点M4、12、16、24、35、48，分别覆盖小M反向约束、已测锚点、4行尾块、完整块边界、11行尾块和端点；M8、13、17、26、38、47留作修正冻结后的前台M验证。已有真实计划已看过结果，以上只是新受控测量的拟合/验证拆分，不能重新宣称为未见真实计划。W13/W2分别保留同route T0和累计DDRC请求测量；W2还需完整expert前序对照。只有通过分辨率与计时扰动门槛的同次调用块级测量才能验证段内分布，跨M累计预算相减仍只能标注为差分假设。背景单独请求率用于独立需求特征；joint总DDRC减BG-only不能自动归属前台，因为背景本身会被前台减速。

混合背景实现前还需解决控制线程亲和性：候选真实布局用满CPU240–319，原协议controller_cpu=240将与计算线程重叠。应选择并记录可用的域外控制核，或者显式减少计算核并承认拓扑不同；不能悄悄沿用原亲和性。当前仅形成下一轮设计约束，尚未改采集器、冻结协议、模型或启动新实验。

覆盖审计保存为receiver_mixed_holdout/request_coverage_diagnostic_20260915.json，SHA256 `ebd957ad085fb391e99ad57060b5a6155cca77fefec3b01d31ddfe453b59c4f8`，包含204个expert/stage条目、14个来源分组及jobs输入哈希。验证了expert身份唯一、segment成本求和守恒、窄计划M1–12预算一致、有限正成本和JSON回读。没有运行新性能测试。


### 固定响应与T0的请求预算扰动（2026-09-15）

本地直接调用既有dynamic_resource_candidate/simulate.predict，对mixed_narrow固定arrival_index=0做5次诊断（合计约42 s）。启动前冻结identity中全部文件哈希匹配；未扰动结果的完整JSON与原预测完全一致，completion=30431.789823 us。只缩放指定1T/M组W13/W2 segment的dram_read_kib、dram_write_kib；T0、请求窗口、gather、队列系数、容量、到达及依赖均固定，模拟器原有请求守恒检查正常通过。不修改jobs、源码或模型参数。

| 人为诊断场景 | 预测计算完成us | 相对原预测变化us | expert203/M16 W13时长us | 同expert W2时长us |
|---|---:|---:|---:|---:|
| 原预算 | 30431.790 | 0 | 1837.232 | 1071.673 |
| 1T/M1–11预算×0.8 | 30181.708 | -250.082 | 1837.232 | 1049.872 |
| 1T/M1–11预算×1.2 | 30681.887 | +250.097 | 1837.232 | 1093.480 |
| 1T/M12–48预算×0.8 | 30299.183 | -132.607 | 1777.377 | 1029.659 |
| 1T/M12–48预算×1.2 | 30586.299 | +154.509 | 1910.834 | 1113.495 |

±20%是事先选定的灵敏度扰动，不是测量置信区间、物理上下界或新校准。仅限这个arrival与局部范围：提高中M预算20%只使目标W13增加73.602 us、整计划增加154.509 us，明显小于已有约2.3 ms的同route W13竞争增量残差和约5.2 ms的整计划偏差尺度。两类实测来自不同样本汇总，不能当作本arrival的逐样本差值；此处仅比较量级。结论是“中M累计请求预算在±20%范围内的统一调整不足以单独消除主要残差”，不能扩大为任意请求量/块分布修正均无效。

下调小M预算20%使该组平均W2预测减少44.255 us，同时让整计划更短，提示修正小M高估和修正整计划低估并非同方向操作。保持独立请求测量优先，但不可据此取消接收敏感度、时间分布和动态混合的验证；没有搜索最优扰动系数，也没有将此诊断作为新留出通过。保存budget_sensitivity_first_arrival_20260915.jsonl及同名metadata.json，记录全部场景、组均值、输入/输出哈希及限制；尚未扩大到31arrival或其他计划。


### 固定预算与T0的请求服务窗口扰动（2026-09-15）

沿用上一节mixed_narrow/arrival_index=0，直接调用既有simulate.predict三次（约24 s），仅把指定1T/M组的W13/W2 segment.dram_service_us乘0.5；T0、累计read/write预算、响应系数、容量、gather和调度输入保持不变。原预测完整JSON再次一致，输入identity匹配，请求守恒检查通过。这个窗口是模拟器的独立DRAM服务时钟长度；缩短后初始名义发出率提高、请求完成后可能被剩余计算时钟遮盖，不等价于直接测得的硬件issue burst。

| 场景 | 整计划预测us | M1–11平均W13/W2时长us | M12–48平均W13/W2时长us | expert203/M16 W13/W2时长us |
|---|---:|---:|---:|---:|
| 原窗口=T0 | 30431.790 | 1276.915 / 741.172 | 2644.658 / 1480.155 | 1837.232 / 1071.673 |
| 仅M1–11窗口×0.5 | 30431.382 | 687.257 / 347.874 | 2676.587 / 1600.099 | 1837.232 / 1170.889 |
| 仅M12–48窗口×0.5 | 30363.105 | 1065.860 / 633.221 | 2491.205 / 1226.120 | 1680.330 / 841.680 |

小M自身预测变化很大但整计划仅变化-0.408 us，且中M/W2反而增加119.945 us：自主时间线、请求集中和关键路径会抵消或转移局部变化。不能以整计划几乎不变证明局部模型准确。缩短中M窗口则让M16预测回到T0，并没有解释实际联合减速偏大；此诊断方向下不能靠减少窗口长度修复低估。0.5不是已拟合值，不能据此采用“小M窗口0.5”，也不证明一般的阶段时序修正无效。

两组灵敏度检查共同要求后续同时验收分组阶段误差与lane/整计划时间。窗口、预算和接收响应必须有独立证据约束，不能只以整计划拟合优劣辨识。结果保存在window_sensitivity_first_arrival_20260915.jsonl与同名metadata.json；尚未扩大到31arrival/其他计划，不是新硬件实验或泛化验收。模型、原jobs和运行中的分配对照均未修改。


### 下一轮独立请求采集器复用边界（2026-09-15）

只读检查narrow_stage_demand/stage_service.cpp、t1/measure.py、request_curve.py及receiver_tail_matched/foreground.cpp，未运行或修改采集器。现有窄team需求采集器的正式grid只有M12/48/96、W13/W2、copies1/32；native虽允许M1，但Python grid的M1是stage=-1的idle控制，不是M1 GEMM需求测量。request_curve明确拒绝M<12，不能从已有正式grid宣称小M需求已经验证。

旧请求native按stage选择独立初始化的a13/a2和B13/B2，W2不会消费刚完成W13的输出；行循环虽顺序执行完整块与尾块，但仅记录整次调用begin/end。现有receiver_tail_matched具备块时长边界，却只接受M16/48/52/76/96/100/112及copies4，同样使用独立a2，且块时长不等于块级DDRC请求量。两个程序都不能直接承担“任意小中M、同expert W13→W2、块请求分布”的完整验证。

因此下一轮实现应分清可复用与需新增的能力：累计独立需求可复用旧DDRC/perf封装和完整调用计数，扩展M grid/native校验并保留M12/48锚点；块时长可复用tail probe的边界检查，但必须通过插桩扰动门槛；W13输出到W2输入的真实连接与阶段计数归属必须独立实现/核验，不能仅在命令顺序上先跑一次W13就声称真实expert前序。copies32的旧需求与copies4的接收探针是不同复用条件，后续匹配实验需显式区分，不能把二者拼成同条件测量。短块的全域DDRC分辨率是否足够仍未知，未通过验证时仅报告累计请求和块时间，不生成虚假的块请求真值。

该源码检查完成了复用范围确认，尚未开始新采集器修改、构建或远端测量；当前allocation_control冻结源保持不变。


### 小/中M独立请求采集器准备（2026-09-15）

E类新Lab receiver_small_middle_demand，登记measurement.small_middle_request。复用narrow_stage_demand的whole-stage计数native和perf封装，native仅扩展M范围到1..96、copies到1/4/32；核函数、参数布局、行循环及生产接口不变。source_origin.json记录旧源文件身份；格式归一化比对确认native仅有输入范围变化。新measure.py使用明确train/validation拆分、37格/组和采集前参数校验，默认窗口500ms，保留输出独占创建和RESET累计时钟差分处理。W2仍是独立A2，明确不宣称真实expert前序或块级请求分布。

本地命令 `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_small_middle_demand/test_protocol.py tmp/joint_cost_model_20260911/receiver_small_middle_demand/test_native.py`：10 passed、8 skipped，0.14s；skip为当前macOS缺少Linux/SVE256目标运行环境。协议测试覆盖训练/验证隔离、M与stage/copies覆盖、非法运行参数、RESET计数和enabled/running差分。native测试已准备，覆盖1..13行kernel、多个完整/尾块边界、三种复用条件及非法cell，但尚未执行。Ruff format/check、clang-format检查、bash -n通过，旧源哈希未变。没有native build、PMU smoke或性能结论。

状态仍为准备中：待补充原始记录完整性/身份/PMU分析器，再在当前allocation批次结束后做Arm correctness、control/PMU smoke、固定native身份和正式运行预算。README保留精确构建/测试/采集语法及这些先决条件。回滚边界仅新Lab目录和本次manifest条目；未修改运行中的allocation目录、jobs、模型公式或默认参数，未提交或启动新远端任务。


### 小/中M请求原始数据验收器（2026-09-15）

receiver_small_middle_demand/analyze.py已完成本地实现。load要求固定native SHA256、完整metadata/complete标记、明确split、逐轮全部37格、唯一记录、CPU/数值检查、有限时间和完整67计数器；重建seed指定的随机采集顺序并逐项匹配。配对要求control无PMU、measurement有PMU且grid/split/seed/rounds/warmup/duration一致；formal模式另要求31/5/500。按各事件自身enabled时间归一化，保留idle扣除原值，检查两组的单计数分辨率<=5%和PMU/control中位时间变化<=5%。输出采集质量与请求拟合资格两个字段；负请求不裁零、不判为可拟合。

命令 `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_small_middle_demand/test_analyze.py`：15 passed（最终含实际随机顺序校验版本0.23s）。合成数据验证可手算的字节/调用结果，以及缺格、重复、未完成、错误身份、越界轮次、CPU失败、无效/缺失计数器、NaN、错误网格和schedule、质量失败、负扣除保留、smoke冒充formal的拒绝路径。这是分析器测试，不是硬件采集证据。README、design状态和manifest同步更新；native与正在运行的allocation批次未改动。


### 独立请求采集器进程链路检查（2026-09-15）

新增receiver_small_middle_demand/test_measure.py，以tmp_path下临时假native驱动真实measure.py，再由真实analyze.load验收。覆盖train和validation两组、随机顺序、warmup排除、READY/ARMED/GO/DONE/ACK握手、错误native身份不能写complete以及旧raw不可覆盖。测试命令 `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_small_middle_demand/test_measure.py`：4 passed，2.25s。假native的时间与数值标志均为合成fixture，不是Arm正确性或性能证据；没有修改采集器行为，也没有运行新远端任务。README和manifest增加复现命令。


### 前台分配与旧新探针对照完成验收（2026-09-16）

通知于00:37:36 UTC记录command_succeeded/exit0。只读核对远端原timeout257182及session2采集器458464均已退出，formal/COMPLETE存在；未重启。rsync取回关闭的两场原始数据和分析产物，在本地独立调用analyze_control与compare_allocation重算。初次Python对象直接比较仅因perturbation.cell的tuple/list序列化类型不同失败；按JSON序列化口径，两场analysis及comparison均完全一致。21个冻结源文件哈希逐项匹配，10个native PID跨场唯一。证据formal/local_parity.json：216项variant、288项场内重复、144项跨场比较完整。

两场均通过既定质量门槛。最大PMU扰动绝对值0.132425%/0.119748%，最大块计时扰动0.203578%/0.211217%，最大单调用计数占比均2.272727%。原始SHA256：session1 `4c4d57a90d4044e3a2f4ebe5b6ceea4993eb3956ce134304da21bd272bbd613b`；session2 `cacf16a5f48aba076ba7a49749376d31e00532171e551c1855a06106a434b562`。完成采集与模型验收仍分开。

配置保持设计：Arm NUMA3，前台CPU319/1T/M100/copies4，独立W13或W2；目标背景M12/W2、每LLC六个4T team、copies32。每次target之前500ms A或B背景前序，A=M12/W2 6+6，B=M48/W13 7+7。old、single、dual三前台actor及两个背景actor同时驻留；single一bank，dual两bank，只有选中的前台活动。每场12正式轮、3 warmup、六轮顺序重复；seed616451/616452。以下整次调用使用block_timing=0、PMU关闭、mean_ns；每轮先joint−solo，再取median，不把块总和当整次调用。

| 前序/探针 | W13竞争增量中位数us 场1/场2 | W2竞争增量中位数us 场1/场2 |
|---|---:|---:|
| A/old | 758.300 / 854.500 | 341.485 / 415.590 |
| A/single | 669.800 / 750.300 | 239.510 / 273.495 |
| A/dual0 | 693.700 / 752.600 | 199.980 / 225.315 |
| A/dual1 | 671.000 / 758.250 | 196.370 / 234.695 |
| B/old | 869.450 / 997.450 | 447.355 / 515.960 |
| B/single | 722.300 / 839.950 | 321.005 / 403.410 |
| B/dual0 | 761.200 / 856.950 | 347.535 / 376.085 |
| B/dual1 | 767.900 / 834.800 | 338.630 / 412.195 |

old→single同轮配对变化如下。三列各自取配对差的中位数，因此中位数不要求相加；逐轮Δjoint=Δsolo+Δincrement恒等式已验证。

| 场/前序/阶段 | Δsolo us | Δjoint us | Δ竞争增量 us |
|---|---:|---:|---:|
| 1/A/W13 | -249.750 | -336.700 | -86.150 |
| 2/A/W13 | -251.350 | -352.800 | -103.600 |
| 1/B/W13 | -252.100 | -401.500 | -149.100 |
| 2/B/W13 | -257.700 | -411.100 | -155.750 |
| 1/A/W2 | -257.440 | -361.630 | -104.275 |
| 2/A/W2 | -271.175 | -412.200 | -140.560 |
| 1/B/W2 | -257.500 | -384.525 | -126.460 |
| 2/B/W2 | -264.985 | -379.770 | -115.130 |

旧新探针之间既有约250–271us的solo基线变化，也有约86–156us的竞争增量变化。不能用旧探针T0配新探针joint来拟合响应。计时kernel代码此前已做等价源审计，但allocator/alignment、驻留集合与地址状态仍混杂；本结果不是生产kernel优化或物理映射因果证明。

同进程dual1−dual0的竞争增量不稳定：B/W2整次调用为-11.355us（场1）、+35.050us（场2）；对应4行尾块为-6.898/+15.267us。A/W13整次为-21.800/+0.800us，尾块为+12.263/-7.275us。不能据此给bank0/1固定标签或修正系数。单bank与双bank还有resident集合/切换差异；本设计缺少“双bank驻留但始终只用一个bank”的独立对照，不能拆出纯切换效应。

所有16个context/stage的整次竞争增量中位数在第二场均增加，范围25.335–128.000us；同进程后六轮减前六轮的配对增量变化中位数最大绝对值11.000us。新分配方案没有消除跨场漂移。跨场仍同时改变进程、地址及运行时间，不能归为单一机制，也不能把这一级别M100诊断替代真实M12–48毫秒级残差。

详细solo/joint/增量及24个逐轮探针对照、32个场内重复记录保存formal/whole_call_diagnostic.json，保留raw哈希和样本；块级对照仍在comparison.json。决定：不采用bank修正、不替换T0、不拟合本批数据；按既定优先级继续小/中M独立需求与接收响应辨识。下一轮新采集器保持原需求probe的分配/执行方式作为明确对照，先做Arm数值及control/PMU smoke；任何新探针都使用自身同条件solo。整计划泛化与等预算planner验收仍未通过。


### 小/中M采集器Arm测试与短测验收（2026-09-16）

allocation结束后，receiver_small_middle_demand在Arm NUMA3构建；OMP_NUM_THREADS=1、OMP_DYNAMIC=FALSE、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1，numactl --physcpubind=240-319 --membind=3。命令为 `.venv/bin/python -m pytest -q` 加该目录的test_native.py、test_protocol.py、test_analyze.py、test_measure.py：37 passed，26.07s，无跳过。记录build.log、build_identity.txt、arm_validation.log；native SHA256为0536b900c928a31e9d8f7058e770740963faa233a6ba838c20e75e15f80e1344。数值证据仍限于保留的常数输入oracle。

随后使用同一binary、train split、seed616460、rounds1、warmup0、duration-ms500，先measure.py --no-pmu，再启用PMU，分析器未加--formal；远端整条命令受10分钟timeout限制。恢复上下文后只读确认无原采集进程且smoke/analysis.json已生成，未重复启动。取回关闭产物，逐项核对smoke/source.sha256，并在本地独立调用analyze.analyze：完整JSON与远端一致。36个GEMM格及idle完整，36格采集门槛通过；最大PMU/control时间差绝对值3.829327%，最大单计数占比0.990099%，idle扣除后无请求拟合资格失败格。证据保存在smoke/local_parity.json及原始control.jsonl、pmu.jsonl。

本次仅一轮、无warmup，formal_protocol=false，不用于拟合或稳定性结论。W2仍为独立A2，未验证真实W13→W2状态迁移。尚未启动31轮正式采集；正式源冻结、运行预算和事件通知启动记录仍待完成。生产模型与参数未改动。

### 小/中M独立需求正式训练采集启动（2026-09-16）

短测与37项Arm测试通过后冻结16个Lab源/协议/运行文件（formal_source.sha256），远端逐项一致。新增run_formal.sh仅编排现有已测试采集器和分析器，bash -n通过；不改采集算法、模型、kernel或生产入口。正式训练M4/12/16/24/35/48、W13/W2、copies1/4/32及idle，seed616461；control与PMU均31轮、5 warmup、500ms，其他亲和性与短测一致。窗口合计1332秒（22.2分钟），外层timeout为60分钟。验证M网格暂未采集，待训练侧方案冻结后使用。

命令由scripts/run_experiment_notify.py launch托管，状态目录receiver_small_middle_demand/notification_train，supervisor_pid83396，thread01a08485-1b77-71a0-b0bd-03fe88a12129；远端命令为 `cd /home/zhangxu/codex/fused_cpp && timeout --signal=TERM --kill-after=10s 60m bash tmp/joint_cost_model_20260911/receiver_small_middle_demand/run_formal.sh`。结果路径同Lab的formal_train/，独占mkdir防重复，保存冻结源、构建身份、起止状态和原始配对数据；只有formal分析器通过才生成COMPLETE。启动不代表采集验收；事件后需核对终态、完整性、独立重算和门槛，不建立模型轮询循环。

### 独立需求训练结果的预先分析口径（2026-09-16）

正式数据尚未查看时保存receiver_small_middle_demand/analysis_plan_v1.md，位于运行中冻结文件集合之外，不改变采集源或门槛。优先检验M4与M12共用预算、M12/M48对M16/24/35的插值误差，以及copies1/4/32复用差异；全部按W13/W2与copies分开。先用本次锚点检验曲线形状，再单列历史锚点跨会话差异，避免把漂移归因于插值。

31轮报告median/MAD、范围及前15/后16轮变化，不把相关轮次当独立会话。小/负write预算保留原值并报告绝对误差。验证采集前冻结两组预测：旧形状假设（小M用M12、其余M12/M48插值）和训练节点相邻线性插值；不插值copies，不在看完留出后增删节点。新表的需求误差改善不能替代联合响应、真实W13→W2状态、未见整计划及等预算planner验收。

### 正式需求离线诊断工具准备（2026-09-16）

新增receiver_small_middle_demand/offline/diagnose.py及test_diagnose.py，未修改formal_source.sha256中任何文件，未读取正式半成品。CLI直接复用既有analyze.analyze(formal=True)验证原始control/PMU流，再计算每格31轮median/MAD、范围、前15/后16变化，保留全部样本、失败资格、raw/native身份。生成M4和中M的逐点粗锚点误差、stage/copies/resource分组摘要、复用对照，以及144项（含read/write与两候选）前瞻预测；每项标明来源节点，不合格来源保留null预测。分组摘要包含M4与三个中M点，诊断时须同时查看逐点结果，不以汇总抵消解释形状误差。历史锚点比较仍待独立核对原bank来源，尚未自动实现。

验证命令 `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_small_middle_demand/offline/test_diagnose.py`：16 passed，0.13s；Ruff format/check通过。可手算fixture覆盖统计量、线性插值与M4共用M12的误差、非正分母、无效样本、短测/validation/缺格/重复拒绝、不合格节点传播、禁止外推。没有新硬件结果或模型改善结论。正式数据关闭并独立验收后，再运行此CLI保存独占预测产物。

### 历史需求锚点可比性核对（2026-09-16）

只读核对narrow_stage_demand/t1/request_bank.json：其引用protocol与两场analysis身份逐项一致，当前旧源仍匹配新Lab的source_origin，counter_metrics.py与linux_perf_event.py新旧字节一致。两者DRAM预算均按各event enabled时间归一化、扣除同轮idle rate、除以测得calls/s，再取MiB/call中位数。共有锚点M12/M48、copies1/32、W13/W2；几何为1T full-owner (1,0,0,1,1)，W13/W2为8/4 MiB，CPU319、controller240、NUMA3。

但历史两场raw metadata的duration_ms=100，M>=48在采集器中提升至500ms；新批全部500ms。因此M12从100ms变为500ms，idle窗口也变化，M48则保持500ms。历史native与新binary身份不同，grid和随机顺序也不同。未来历史锚点差值只能标记跨协议/会话差异，不可仅归因于地址漂移或M插值。优先使用新批内部M12/M48检验形状；不为这个诊断重跑历史实验。核对记录offline/historical_comparability.json；未读取正式训练观测。

### 离线预测CLI链路补充验证（2026-09-16）

新增一项真实子进程CLI测试：复用已有合成raw fixture扩展为完整31/5/500随机顺序，经过真实analyze加载与diagnose CLI，验证144条预测的可手算read/write值和raw身份；第二次向同一输出运行必须FileExistsError且原内容不变。仅新增offline测试，无采集源或分析算法变化。命令 `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_small_middle_demand/offline/test_diagnose.py`：17 passed，0.52s，Ruff通过。合成数据不证明硬件或模型泛化，正式训练观测仍未读取。

### 小/中M正式训练需求完成验收与插值诊断（2026-09-16）

监督进程退出事件到达后核对completion：command_succeeded/0，01:20:36 UTC，通知accepted。远端正式产物started_utc=00:49:01Z、COMPLETE=01:16:44Z，采集/分析约27分43秒；取回关闭产物，exit_code=0，16个冻结源身份匹配。本地独立analyze(formal=True)与远端analysis完整一致；36格均通过既定采集门槛，最大PMU/control差绝对值3.262732%，最大单计数占比0.990099%，无请求拟合资格失败格。原始control SHA256 508dc4fac0b66f22883bdbbbd8eea1244ab0c5bb77b6bc935aa83ba98e72192d；PMU d49625623d5c5d9586ef23f1ae39010122ca0907d686c705b8744c3c04d25292。formal_train/local_parity.json保存独立验收。

本批Arm 1T CPU319/controller240/NUMA3、full-owner W13/W2=8/4MiB、独立A2，训练M4/12/16/24/35/48、copies1/4/32、31轮5warmup500ms、seed616461。固定工具offline/diagnose.py从完整raw生成diagnostic_predictions_v1.json，不改变原分析或模型。以下为copies32的整阶段idle扣除读请求中位数，单位MiB/call：

| 阶段 | M4 | M12 | M16 | M24 | M35 | M48 |
|---|---:|---:|---:|---:|---:|---:|
| W13 | 7.6967 | 7.6527 | 14.7629 | 14.8440 | 18.4543 | 19.3195 |
| W2 | 3.7106 | 3.7151 | 6.0658 | 6.1330 | 8.0040 | 9.0686 |

用同批M12/M48锚点线性插值，M16/W13低估5.8139MiB（39.38%），M16/W2低估1.7559MiB（28.95%）；M24分别低估22.25%/10.33%，M35低估18.14%/10.85%。M16在copies1/4下同样低估：W13为24.48%/33.77%，W2为29.29%/22.28%。M16与M24接近，提示整阶段累计需求在增加完整块/尾块附近存在明显非线性；这不是对块级请求的直接测量，也不是证明全部运行时低估由此造成。

M4共用M12的假设依赖复用状态：copies32下读预算误差W13 -0.57%、W2 +0.12%；copies4为+5.39%/+8.69%；copies1为+30.63%/+36.10%。不能从copies32这一对点推广为全部小M或复用环境统一预算。

采集门槛通过不等于稳定性通过。copies1/W13的M48读请求median9.7014MiB、MAD3.4732MiB、后16轮减前15轮中位数-5.1866MiB；M16对应median3.8368、MAD1.0751、前后差-1.3030MiB。该组明显漂移，需要独立会话复测，不能直接采用本次中位数为稳定校准。write中位数范围0.0000531–0.0582292MiB/call，须保留绝对误差与原样本，避免近零百分比主导结论。

两种既定候选的144条验证预测已保存并用prediction_freeze_v1.json冻结；验证数据尚未采集。当前结论为旧M插值存在需求层缺陷，同时部分复用条件有稳定性问题。未更改T0、竞争响应、planner或生产模型；下一步按冻结预测做M留出对照，并另行验证漂移组重复性，不能以需求表通过代替联合模型验收。

### 未见M需求验证集启动（2026-09-16）

notification_train用户消息对应已验收的01:20:36 UTC同一终态，未重启或重复分析训练采集。校验已冻结预测SHA256 9e0ef3787a340c1b72982c2db837b490074f38d6b12ff509394fcb1a56dc1e79与analysis_plan身份后，新增run_validation.sh；仅把既有编排切换为validation、seed616462、formal_validation独占目录，并增加冻结预测/脚本校验。bash -n通过，沿用已有Arm37项、离线17项验证，不重复无关测试。原采集16文件与模型不变。

验证M8/13/17/26/38/47，W13/W2、copies1/4/32、31轮5warmup500ms，CPU319/controller240/NUMA3，60分钟远端timeout。远端预测身份逐项通过、旧采集进程不存在后，通过scripts/run_experiment_notify.py launch启动，监督PID85645，状态目录notification_validation，结果formal_validation/；命令 `cd /home/zhangxu/codex/fused_cpp && timeout --signal=TERM --kill-after=10s 60m bash tmp/joint_cost_model_20260911/receiver_small_middle_demand/run_validation.sh`。验证用于比较两组预先冻结的插值，不重拟合既有预测；copies1/W13漂移的独立重复仍待做，需求留出不等于联合环境/真实expert/planner验收。结束后按事件续接独立验收，不建立模型轮询循环。

### 留出需求评估工具准备（2026-09-16）

新增offline/evaluate.py及test_evaluate.py，未修改运行中冻结源。CLI首先要求预测文件精确匹配验证启动前SHA256 9e0ef3787a340c1b72982c2db837b490074f38d6b12ff509394fcb1a56dc1e79，再复用analyze(formal=True)从完整验证raw重算。核对native、36格与144条预测完整唯一，输出逐点MiB误差、正参考值相对误差，以及stage/copies/resource/candidate分组bias/MAE/MAPE。失败点全部保留，分组明确total/eligible/relative_count/complete，近零或非正参考不伪造百分比。没有拟合入口，也没有把需求MAPE当成整计划验收。

命令 `.venv/bin/python -m pytest -q tmp/joint_cost_model_20260911/receiver_small_middle_demand/offline/test_evaluate.py`：10 passed，0.14s；Ruff format/check通过。合成可手算的-20%/+10%例子覆盖误差符号、zero-write处理、不合格点保留，以及split/短测/native/缺格/重复/预测篡改拒绝。未读取正在采集的验证数据；尚无留出效果结论。

### copies1/W13漂移通道核对（2026-09-16）

只使用已关闭训练数据，按预先规定的前15/后16轮取中位数，保存formal_train/drift_channels.json，覆盖全部36格的读请求、PMU service、control service和idle扣除L2 refill/调用。copies1/M48/W13：DRAM read为13.941664→8.755047MiB/call（-37.20%），PMU service4930.790→4933.030us（+0.0454%），L2 refill489628.3→494201.2 events/call（+0.934%）；control service4931.310→4931.925us。M16/W13 read4.927693→3.624667MiB/call（-26.44%），service1677.320→1668.650us（-0.517%），refill235889.2→235454.7（-0.184%），control1660.200→1660.240us。

未扣idle的M48 read同样13.972441→8.788072MiB/call，M16为4.938626→3.635692；对应idle读扣除最大仅0.062607/0.021078MiB/call，不能解释5.1866/1.3030MiB变化。全局idle读速率前后中位数0.006628→0.006811GB/s。copies1/W2六个M的前后读请求均增加，与W13六个M均下降方向不同，因此不支持直接乘一个全局请求漂移系数。

结论限于通道分离：T0及L2 refill近似稳定并不保证DRAM请求稳定，不能以无竞争时间稳定作为外部压力稳定的充分条件。可能涉及L2之后的服务来源/复用状态，但当前全域DDRC与核心事件无法判定具体LLC机制；未做因果归因、系数拟合或模型改动。独立会话及受控前序复测仍需保留。运行中的验证集未读取。

### 未见M验证终态恢复与冻结预测验收（2026-09-16）

notification_validation于01:46:36 UTC返回SSH255，command.log为server192.168.29.64不响应；首次只读重连banner超时。随后同alias重连成功，原PID677037及该Lab所有采集/运行进程均已不存在，远端exit_code=0、COMPLETE=01:48:29Z。实验在SSH断线后继续完成；未重启。断线具体原因未判定。取回关闭产物后，formal_source/validation_source身份全部匹配，本地analyze(formal=True)与远端analysis完整一致。36格全部通过采集门槛，最大PMU/control扰动3.085991%，单计数占比最大0.990099%，无资格失败格。证据formal_validation/local_parity.json。raw control SHA256 e73b034f569c44ef79f05ef0abe6a45aeccf9b69fcadb26a7f76b39820bf936e；PMU b79163ee9a5f5ae018e50269f2754d7436177857ace51617baf162307c9f38cd。

使用offline/evaluate.py对启动前冻结的144条预测验收，无拟合或预测修改；输出formal_validation/holdout_evaluation_v1.json。协议M8/13/17/26/38/47，W13/W2、copies1/4/32、31轮5warmup500ms、seed616462，机器/几何/native同训练。下表为6个未见M的DRAM读请求MAPE，非运行时间误差：

| 阶段/copies | 粗锚点MAPE % | 相邻节点MAPE % |
|---|---:|---:|
| W13/1 | 31.656 | 31.254 |
| W13/4 | 12.687 | 5.708 |
| W13/32 | 22.008 | 8.615 |
| W2/1 | 17.479 | 15.959 |
| W2/4 | 20.813 | 17.159 |
| W2/32 | 18.306 | 9.913 |

相邻节点平均误差改善但仍有系统性块边界低估。copies32下，相邻插值M13/W13低估34.61%、W2低估28.18%；M17为-0.92%/-0.25%；M26为-11.65%/-18.44%；M38为-2.20%/-11.37%；M47为-1.72%/-0.76%。M13实测W13 14.422MiB/W2 5.991MiB，接近训练M16的14.763/6.066；从M12到M16作线性插值将新增块的增长摊平。copies4/W2的M13/26/38也分别低估36.07%/27.46%/21.92%，其请求MAD仅约0.026/0.060/0.105MiB，不能笼统当成随机波动。

因此两种平滑M插值均未形成可靠泛化需求模型，不采用新表或把留出结果反拟合回冻结版本。下一候选须显式考虑完整块数/尾块及此前B访问历史；当前验证集已看过，只能作开发诊断，新候选需要另选前瞻点。copies1/W13跨状态/会话误差与块边界结构问题分开验证，保留独立重复需求；单纯添加全局倍率不成立。此次未评估联合时间或planner，未更改T0、竞争响应和生产参数。完整144条读写误差、样本、分组资格均保留在评估JSON，write不因近零分母省略。

### 块数结构回顾性对照与可辨识缺口（2026-09-16）

在已看过的验证集上作明确post-hoc诊断：按ceil(M/12)用训练M12/24/35/48的同stage/copies累计预算预测，不拟合系数。第三组用M35代理缺失M36，且假设同块数下尾长无影响，因此不可作为正式模型或新留出通过。结果offline/block_count_diagnostic_v1.json包含全部72条读写误差及来源身份。

读请求MAPE按W13 copies1/4/32为14.531%/14.119%/1.919%，W2为15.286%/3.629%/1.196%。copies32显著优于平滑插值；copies4/W13则从相邻插值5.708%恶化为14.119%，M13高估38.00%。块边界结构必要，但不能普遍丢弃尾块/复用状态。copies1漂移也未解决。这些是候选辨识证据，不是生产性能改善。

新增offline/block_model_next_design.md明确D(M)=完整块累计项+按此前完整块数和r条件化的尾项，指出现有train缺M36及多个history下的尾长对照。提出每history采r1/4/8/12、两独立会话，并在预测冻结后使用新的r留出；预计总201分钟、拟定硬超时总220分钟。尚未修改采集器、拟合物理项或启动新实验，独立DDRC累计差不冒充块级计数真值。旧验证集从此仅为开发诊断。
