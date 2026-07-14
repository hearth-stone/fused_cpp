# CPU MoE 调度数学模型

> 状态：调度问题定义的 source of truth。
>
> 最后更新：2026-07-13。
>
> 修改 planner 的决策变量、目标函数、资源约束、线程宽度集合、调度语义、
> rank 耦合方式或剪枝策略时，必须同步更新本文档及末尾变更记录。

## 1. 文档边界

本文档将 MoE 调度分为四层：

1. **原始数学问题**：定义完整可行域，不包含当前实现的经验剪枝。
2. **实例参数**：kernel/cost model 提供执行时间和资源需求。
3. **求解器编码**：CP-SAT、MILP 或启发式对同一个问题的不同表达。
4. **工程剪枝**：线程宽度、静态 shape、LPT、non-idling 等当前限制。

M12、SVE、W13 split、具体权重布局等属于实例参数生成过程，不属于核心问题
定义。核心问题只观察它们最终产生的执行时间和资源需求。

## 2. 原始问题

### 2.1 集合与参数

设一次 rank-local MoE 执行包含 expert job 集合：

$$
\mathcal J=\{1,\ldots,n\}.
$$

可用的最大 CPU 核心/线程数为：

$$
T_{\max}=C.
$$

原始问题允许每个 job 使用任意整数线程宽度：

$$
\mathcal T_i^{(0)}=\{1,2,\ldots,T_{\max}\}.
$$

对 expert $i$ 和线程数 $t$，实例参数生成器提供：

$$
p_i(t)>0,
$$

其中 $p_i(t)$ 是 job 非抢占执行时的处理时间。它可以依赖 route 数和
kernel 实现，但这些依赖不进入核心调度公式。

设 $\mathcal R$ 为 CPU 之外的 cumulative resource 集合。对资源
$r\in\mathcal R$：

$$
d_{ir}(t)\ge0
$$

表示 expert $i$ 使用 $t$ 个线程时的资源需求，$B_r$ 表示资源容量。典型资源
可以是 LLC 驻留工作集或保守预留的内存带宽。若某类性能影响已经包含在
$p_i(t)$ 中，则不应在 $d_{ir}(t)$ 中重复计算。

可选的 precedence 集合为：

$$
\mathcal E\subseteq\mathcal J\times\mathcal J.
$$

当前将完整 expert 抽象为一个 job 时，通常有 $\mathcal E=\varnothing$；若将
expert 展开成通用 phase DAG，则使用 $\mathcal E$ 表示 phase 依赖。

### 2.2 决策变量

每个 expert 选择一个整数线程宽度和开始时间：

$$
t_i\in\{1,\ldots,T_{\max}\},\qquad s_i\ge0.
$$

线程数在 job 开始前选择，执行过程中保持不变。因此 job 是 **moldable**，
不是执行中可改变线程数的 malleable job。

完成时间为：

$$
f_i=s_i+p_i(t_i).
$$

定义非抢占式活跃指示函数：

$$
a_i(\tau)=\mathbf 1[s_i\le\tau<f_i].
$$

### 2.3 资源约束

任意时刻的 CPU 使用量不能超过 $T_{\max}$：

$$
\sum_{i\in\mathcal J}t_i a_i(\tau)\le T_{\max},
\qquad\forall\tau.
$$

其他 cumulative resource 满足：

$$
\sum_{i\in\mathcal J}d_{ir}(t_i)a_i(\tau)\le B_r,
\qquad\forall r\in\mathcal R,\ \forall\tau.
$$

前驱约束为：

$$
s_j\ge f_i,
\qquad\forall(i,j)\in\mathcal E.
$$

### 2.4 目标函数

Rank-local compute makespan 为：

$$
C_{\max}=\max_{i\in\mathcal J}f_i.
$$

原始优化问题为：

$$
\boxed{
\begin{aligned}
\min_{\{s_i,t_i\}}\quad
& C_{\max}\\
\mathrm{s.t.}\quad
& t_i\in\{1,\ldots,T_{\max}\}, &&\forall i,\\
& f_i=s_i+p_i(t_i), &&\forall i,\\
& \sum_i t_i a_i(\tau)\le T_{\max}, &&\forall\tau,\\
& \sum_i d_{ir}(t_i)a_i(\tau)\le B_r,
&&\forall r,\tau,\\
& s_j\ge f_i, &&\forall(i,j)\in\mathcal E,\\
& s_i\ge0. &&
\end{aligned}}
$$

在运行时比较不同 planner 时，还需要计入 planning latency：

$$
T_{\mathrm{runtime}}=T_{\mathrm{plan}}+C_{\max}.
$$

Dispatch、communication、scatter 和 combine 在不受 plan 影响时作为外部成本；
若其成本会随 plan 改变，则必须提升为目标函数的一部分。

## 3. 多 Rank 外层

对 rank $k\in\mathcal K$，定义 rank-local 完成时间 $C_k$。不考虑跨 rank
执行期争用时：

$$
C_{\mathrm{compute}}=\max_{k\in\mathcal K}C_k.
$$

完整 layer 目标为：

$$
T_{\mathrm{layer}}
=T_{\mathrm{dispatch}}
+\max_k C_k
+T_{\mathrm{communication}}
+T_{\mathrm{scatter}}
+T_{\mathrm{combine}}.
$$

若 rank 共享内存带宽等资源，则应在所有 rank 的联合活跃集合上增加共享资源
约束。一个 rank 完成后，联合活跃集合随之改变，剩余 rank 必须切换到新的资源
容量，而不能将 dual-rank 速率应用到整个执行区间。

## 4. 问题类型与复杂度

该问题可描述为：

$$
P\mid\mathrm{moldable,\ nonpreemptive,\ cumulative\ resources,\ prec}
\mid C_{\max}.
$$

这只是描述性记号，不声称存在覆盖全部扩展项的单一标准 Graham notation。

即使固定 $t_i=1$、移除额外资源和 precedence，问题也包含经典
$P\parallel C_{\max}$。因此：

- 优化版本是 NP-hard；
- 有理数离散实例的判定版本是 NP-complete；
- 加入可变线程宽度和 cumulative resources 不会降低复杂度。

## 5. Kernel Implementation Variant

Split/no-split 以及未来其他 kernel 版本不属于核心问题变量。一个全局
implementation variant $v\in\mathcal V$ 只负责生成实例参数：

$$
\Theta_v=\{p_i^{(v)}(t),d_{ir}^{(v)}(t)\}.
$$

对每个 variant 求解同一个调度问题：

$$
C^*(v)=\min_{s,t}C_{\max}(s,t;\Theta_v),
$$

再进行外层选择：

$$
C^*=\min_{v\in\mathcal V}C^*(v).
$$

因此 variant 只改变问题实例，不改变问题类型。若两个 variant 在所有相关
线程宽度上同时具有不更大的时间和资源需求，则被支配的 variant 可以在进入
planner 前删除。

## 6. 求解器编码

### 6.1 数学定义

核心定义直接使用整数变量 $t_i$。这保持问题表达简洁，不绑定具体求解器。

### 6.2 CP-SAT

CP-SAT 可以为每个 $(i,t)$ 建立 optional interval，并要求每个 expert 恰好
选择一个线程宽度。CPU 和 LLC 使用 cumulative constraints。CP-SAT 适合作为
离线、未剪枝问题的 oracle，并返回可行上界和理论下界。

### 6.3 MILP

标准 MILP 通常将 $t_i$ 展开为 one-hot 变量：

$$
y_{it}\in\{0,1\},\qquad
\sum_{t=1}^{T_{\max}}y_{it}=1.
$$

然后线性表示：

$$
t_i=\sum_t t\,y_{it},\qquad
p_i=\sum_t p_i(t)y_{it}.
$$

One-hot 是等价求解器编码，不是线程宽度剪枝。

## 7. 当前实现相对原始问题的剪枝

| 层级 | 原始可行域 | 当前限制 | 性质 |
| --- | --- | --- | --- |
| 线程宽度 | $1,2,\ldots,T_{\max}$ | `1,2,4,8,16,32` | 离散宽度剪枝 |
| 并发配置 | 活跃 job 可形成任意宽度组合 | 整次调用使用静态 core shape | static-partition 剪枝 |
| Shape 集合 | 所有满足容量的整数组合 | profile 支持并经工作集筛选的 shape | 候选剪枝 |
| Assignment | 任意 expert-to-resource 调度 | 按 isolated cost 的 LPT | 启发式分配 |
| Ordering | 任意可行开始时间和顺序 | 每个 lane 的 LPT 顺序 | 顺序剪枝 |
| Idling | 允许主动等待以避开争用 | dependency ready 后立即启动 | non-idling 剪枝 |
| Kernel variant | 任意未被支配的实现 | 当前 split/no-split profile pair | 实例候选限制 |

当前 `IntervalPlanner` 搜索的是上述剪枝后 plan space 中的方案，不是原始问题
的全局最优方案。

## 8. 当前 Cost Model 的位置

原始问题使用固定的 $p_i(t)$ 和硬 cumulative resource 约束。当前工程实现还
使用 stage-aware event simulator，让执行速率依赖并发 active set。令
$x_i(\tau)\in[0,1]$ 表示 job 完成比例，则该扩展可以写成：

$$
\frac{dx_i(\tau)}{d\tau}
=
\frac{a_i(\tau)}
{p_i(t_i)D_i(\mathcal A(\tau))},
\qquad
x_i(s_i)=0,\quad x_i(f_i)=1.
$$

新生成且携带 `iso_formula` 的 schema-v2 profile 使用：

$$
p_i(t)=T_{\mathrm{iso}}(R_i,t)=O(t)+C(R_i)\phi_{\mathrm{USL}}(t)k_\phi(t),
$$

$$
O(t)=o_0+\frac{o_1}{t},\qquad
\phi_{\mathrm{USL}}(t)=\frac{1+\alpha(t-1)+\beta t(t-1)}{t}.
$$

$C(R)$ 是单线程 route-work 的一维实测校准曲线；$O(t)$ 和
$\phi_{\mathrm{USL}}(t)$ 分别表示固定开销与相对理想 $1/t$ 的线程效率基线，
$k_\phi(t)$ 是与 route 无关的一维实测校正。M1/M2/M4/M8 与前两个
M12 panel 的启动/线程效率不同于 steady-state M12 bulk，因此保留小 M 的
`(tail, threads)` 实测 residual；更大的 M 使用公式，并按 kernel 组合规则
追加 tail。公式仅在 profile 的已校准线程域内有效，不承担跨机器或更高线程数
外推。没有序列化公式的历史 profile 继续使用旧二维 table，避免离线重拟合改变
既有 planner 基准；可显式指定 `iso_mode="formula"` 做 shadow 对比。

这是 **scheduling with interference** 扩展，经典固定 processing-time 的近似
保证不再直接成立。推荐的求解分层是：

1. 使用硬 CPU/LLC 约束的 CP-SAT 生成离线 oracle 或候选；
2. 使用 event simulator 对少量候选重新评分；
3. 在线 planner 使用经 oracle 验证过的剪枝和低开销启发式。

## 9. 剪枝验证

令未剪枝 oracle 的最优值为 $C^*$，求解器下界为 $LB$，剪枝 planner 的可行
结果为 $C_{\mathrm{pruned}}$。则：

$$
0\le
\frac{C_{\mathrm{pruned}}-C^*}{C^*}
\le
\frac{C_{\mathrm{pruned}}-LB}{LB}.
$$

每一层剪枝都应分别报告：

- plan-space 缩减量；
- planning latency；
- 相对 oracle 的实际 regret；
- oracle 未证明最优时，由 $LB$ 得到的保守 regret 上界；
- 在真实 event simulator/runtime 上的 held-out regret。

## 10. 同步规则

发生以下任一变化时，必须同步更新本文档：

- job 粒度从 expert 改为 phase，或反向合并；
- 线程宽度定义、最大线程数或可行宽度集合变化；
- 是否允许抢占、迁移、动态调整线程数或主动 idling；
- CPU、LLC、L3/DRAM 带宽等资源约束变化；
- rank-local 与跨 rank 资源耦合方式变化；
- 目标函数加入或移除 planning、dispatch、communication、scatter、combine；
- kernel variant 从全局选择改为 per-expert 选择；
- shape、assignment、ordering 或 cache 相关剪枝变化；
- CP-SAT/MILP 编码改变原始可行域，而不只是等价重写。

同步修改至少应包括：

1. 本文档的公式和剪枝表；
2. 对应 planner/cost-model 测试；
3. oracle/regret 或性能验证结果；
4. 下方变更记录。

## 11. 变更记录

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-07-13 | v0.1 | 建立未剪枝通用问题；将线程宽度定义为 $1\ldots T_{\max}$；区分 kernel variant、求解器编码和当前工程剪枝。 |
| 2026-07-14 | v0.2 | schema-v2 的 isolated processing time 改为 $O(t)+C(R)\phi_{\mathrm{USL}}(t)k_\phi(t)$；保留小 M residual 和二维 table 回归模式。 |
