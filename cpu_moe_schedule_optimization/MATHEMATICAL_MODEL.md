# CPU MoE 调度数学模型

> Runtime integration note (2026-08-14): machine calibration remains an
> explicit deployment action. An installed `MoePlannerRuntime` binds one
> analytical model and cached `PlannedMoE` to the calibration's ordered CPU
> rank. The normal BF16 tiled entrypoint lowers compatible standalone/TP SVE
> fused-SiLU calls to Plan V2; calls outside that domain preserve the original
> dispatcher. The first production version uses a bounded homogeneous-team LPT
> search described below; the full analytical candidate space remains available
> offline. The serialized Plan V2 schema is unchanged.

> 状态：调度问题定义的 source of truth。
>
> 最后更新：2026-08-16。
>
> 修改 planner 的决策变量、目标函数、硬约束、性能响应、线程宽度集合、调度语义、
> rank 耦合方式或剪枝策略时，必须同步更新本文档及末尾变更记录。

## 1. 文档边界

本文档将 MoE 调度分为四层：

1. **原始数学问题**：由 expert route tasks、CPU 容量、真实 isolated time 和
   contention slowdown 定义完整可行域与目标，不包含当前实现细节或经验剪枝。
2. **性能近似**：cost model 估计原始问题中的真实 isolated time 和 contention
   slowdown。
3. **求解器编码**：CP-SAT、MILP、event simulation 或启发式对同一个问题的
   不同近似与表达。
4. **工程剪枝**：线程宽度、静态 shape、LPT、non-idling 等当前限制。

M12、SVE、stage range 数、具体权重布局和 cache 工作集公式不属于核心问题定义。
它们只影响固定执行环境下观测到的 isolated time 和 contention slowdown。

### 1.1 Production full-N 与 tile-window 不变量

当前 ARM SVE production runtime 不再有 legacy weight split、range-count 或
byte-window 决策。每个 whole-expert task 的 W13 和 W2 都覆盖完整 packed-N domain；
Plan V2 可在已选 team width 之外逐 stage 携带 per-worker `window_tiles`，其中 0 表示
full owner stripe。窗口只改变完整 N domain 的访问顺序，不改变权重子集、算术量、
packed layout 或输出 ownership。

对 stage $s=(K_s,N_s)$、BF16 N tile $\nu$ 和 team width $t$，定义：

$$
q_s=\frac{N_s}{\nu},\qquad
a_s(t)=\min(t,q_s),
$$

$$
B_s=2K_sN_s,\qquad
b_s=2K_s\nu.
$$

令每线程窗口为 $w_s$ 个 tile，则

$$
g_s=t w_s,\qquad
R_s=\left\lceil\frac{q_s}{g_s}\right\rceil,
$$

且 $w_s=\lceil q_s/t\rceil$ 即 full-stripe、$R_s=1$ 端点。

$B_s$ 是该 expert 的完整 packed-B stage 字节数；$w_sb_s$ 是最忙 worker 在一个
window 内的 tile-aligned owner footprint。当前 fused expert 中：

$$
(K_{13},N_{13})=(H,2F),\quad B_{13}=4HF,
$$

$$
(K_2,N_2)=(F,H),\quad B_2=2HF.
$$

因此当前计算模式由 $(t,w_{13},w_2)$ 决定。增加 $t$ 或减小 $w_s$ 都会缩小每线程
瞬时 B footprint；只有减小 $w_s$ 会增加 $R_s$，并可能引入跨 window 的 A 重扫与
range restart。production 在 width 选定后使用确定性 band policy 给出窗口，不把
$w_s$ 扩成 planner 自由搜索变量；9.36 的解析 v6 只作 shadow selector。本文后面明确
标为“历史”的 range-count/split 章节仅保留 provenance，不定义当前 ABI。

## 2. 原始问题

### 2.1 Expert route tasks

设一次完整 MoE layer 有 $E$ 个候选 expert：

$$
\mathcal X=\{1,\ldots,E\}.
$$

当前目标模型中 $E=256$。以下公式描述一个调度域：node-level 调度时该域可包含
全部 256 个 expert；rank-local 调度时，$\mathcal X$ 替换为分配到该 rank 的
expert 子集。复杂度分析将 $E$ 视为可变输入规模；固定的单个 256-expert 实例
本身不构成渐近复杂度问题。

若输入是由 $T$ 个 token、无重复 expert 的 TopK router 产生的全局 histogram，
则还满足：

$$
\sum_{i=1}^{E}M_i=T K,\qquad
0\le M_i\le T,\qquad
|\{i\mid M_i>0\}|\ge K.
$$

rank-local histogram 只包含 dispatch 到该 rank 的 routes，因此其 route 总和
不必等于 $TK$；但每个 local expert 仍满足 $M_i\le T$。下文核心调度问题同时
允许全局和 rank-local 输入，不把 $TK$ 等式作为 planner API 的额外硬约束。

路由完成后，expert $i$ 得到 route tensor：

$$
\mathbf X_i\in\mathbb R^{M_i\times H},
$$

其中 $M_i$ 是 route 数。需要执行的 job 集合为：

$$
\mathcal J=\{i\in\mathcal X\mid M_i>0\}.
$$

Job $i$ 表示在其独立权重上完成整个 expert 计算：

$$
\mathrm{W13}_i
\rightarrow\mathrm{activation}_i
\rightarrow\mathrm{W2}_i.
$$

不同 expert job 之间没有数据依赖。route tensor 的数值和 token index 不影响
rank-local expert compute 调度；在 expert shape 固定时，调度只需要 expert
identity 和 $M_i$。若 scatter/combine 也进入目标，token index 才需要进入外层
系统模型。

### 2.2 CPU 容量与真实执行响应

可用的最大 CPU 核心/线程数为：

$$
T_{\max}=C.
$$

原始问题允许每个 job 使用任意整数线程宽度：

$$
\mathcal T_i^{(0)}=\{1,2,\ldots,C\}.
$$

固定目标机器及其执行环境，但不展开 ISA、kernel、layout 或 cache 细节。定义
真实 isolated time：

$$
0<I_i(t)<\infty,
$$

表示 job $i$ 使用 $t$ 个线程且没有其他 expert 干扰时，完成整个 expert 计算
所需的时间。相同 expert shape 下可写成：

$$
I_i(t)=I(M_i,t).
$$

对时刻 $\tau$ 的活跃配置，定义：

$$
\mathcal Z(\tau)
=\{(j,M_j,t_j)\mid j\text{ 在 }\tau\text{ 时刻活跃}\}.
$$

真实 contention slowdown factor 为：

$$
1\le D_i(\mathcal Z) = 1+\rho_i(\mathcal Z)<\infty,
$$

其中 $\rho_i$ 是相对 isolated time 的延长比例，并满足单独运行时：

$$
D_i(\{(i,M_i,t_i)\})=1.
$$

只按并发 expert 数建模是特例
$D_i(\mathcal Z)=D(M_i,t_i,|\mathcal Z|)$。通用定义保留完整
$(M_j,t_j)$ 活跃配置，因为相同并发数下，不同 route 和线程宽度可以产生不同
争用。

基础模型假设 $D_i$ 对 normalized progress 做乘性缩放，并且只依赖当前
$\mathcal Z(\tau)$，即相同 active configuration 的瞬时速率与到达历史无关。
若验证发现 W13/W2 phase、cache residency 或先前执行顺序仍会改变速率，应将
这些状态加入 $\mathcal Z$，而不是修改调度目标。

因此，在 expert shape 和固定执行环境已确定时，原始问题的最小充分输入为：

$$
\boxed{
\mathcal I
=\left(
\{(i,\mathbf X_i,M_i)\}_{i=1}^{E},
C,
\{I_i(t)\},
\{D_i(\mathcal Z)\}
\right).
}
$$

route tasks 与 $C$ 定义工作负载和硬可行域；$I$ 与 $D$ 定义目标机器上的真实
执行时间。后续 cost model 只负责近似 $I,D$，planner 决定线程宽度和开始时间。

### 2.3 决策变量与执行动态

每个 expert 选择一个线程宽度和开始时间：

$$
t_i\in\{1,\ldots,C\},\qquad s_i\ge0.
$$

基础 production 问题在线程开始前选择宽度，执行过程中保持不变。因此
strict/tail-pool job 是 **moldable**，不是执行中可改变线程数的 malleable
job。Job 一旦开始就连续执行至完成，不能抢占。下文 2.3.1 定义 production
planner 在尾 task 启动前做一次宽度选择的 bounded moldable 扩展；2.3.2
定义保持宽度不变、仅迁移未启动完整 expert 的实验性尾部领取；2.3.3
记录已关闭的短 expert 重组与 residual-M suffix 候选；2.3.4 单独保留已退役的
W13/W2 边界伸缩历史模型。

Plan V2 在表示层为每个 task 增加离散允许宽度集合 $\mathcal A_i$、首选宽度
$p_i$、已选执行宽度 $\bar t_i$ 和 placement $q_i$：

$$
\bar t_i=\texttt{task\_threads}[i],\qquad
p_i\in\mathcal A_i,\qquad
\bar t_i\in\mathcal A_i,\qquad
q_i\in\{\mathrm{fixed},\mathrm{tail\_pool}\}.
$$

strict 候选生成：

$$
\mathcal A_i=\{p_i\}=\{\bar t_i\},
\qquad \mathcal R_i=\varnothing,\qquad q_i=\mathrm{fixed},
$$

其中 $\mathcal R_i$ 是合法 resize point 集合。strict runtime 校验 V2 后仍按
$\bar t_i$ 执行 fixed-interval async DAG。production planner 还会搜索
whole-expert `tail_pool` 候选；该模式允许
$q_i=\mathrm{tail\_pool}$ 的完整 expert 在运行时由任一已释放、宽度为
$\bar t_i$ 的对齐线程组领取；线程组只有在覆盖它的全部 fixed tasks 完成后才
释放。pooled task 无前驱，fixed task 也不能依赖 pooled task。即使外部 V2
携带更宽的 $\mathcal A_i$，当前 executor 也不会在 task 执行中选择其他宽度。
因此 tail pool 只扩大 **expert 边界处** 的 resource assignment/start-time
可行域，不把 job 从 moldable 改成 malleable，也不改变本节后续公式中的
$t_i$。

当前自动候选仍是受剪枝的 threshold policy。给定上界
$R_{\max}=12$，候选 threshold 来自与 route signature 相同的稳定 bucket：

$$
\mathcal R_{\mathrm{pool}}
=\{r\in\{1,2,4,8,12\}\mid r\le R_{\max},
\ \exists i:M_i\le r\}.
$$

产生相同 pooled expert 集合的 threshold 只保留最小者。候选 threshold 使用
固定 bucket；shape/strategy cache 除原 route signature 外，还记录每个候选
threshold 下的 eligible expert 数。因此 expert ID 或不跨 threshold 的 routing
抖动仍可复用计划，而任一 pooling boundary 的参与数量变化都会重新搜索。

对 $R\in\mathcal R_{\mathrm{pool}}$，pooled expert 集合和 fixed 集合分别为：

$$
\mathcal P(R)=\{i\mid M_i\le R\},\qquad
\mathcal F(R)=\mathcal X\setminus\mathcal P(R).
$$

自动 pool width $g$ 从当前 cost model 可计算的 $\{1,2,4\}$、能整除 $C$，且与
全部 fixed interval 对齐的离散宽度中选择；显式 forced override 仍允许其他已
校准宽度。dynamic 候选只在 strict 最优 uncertainty band 内的 head shape 与
strict point estimate 最快的两个 head shape 上展开。planner 不预先决定真实
领取时刻，而是构造确定性的 list-scheduling surrogate 来评分。令
$\mathcal B_h$ 为覆盖第 $h$ 个 $g$-thread group 的 fixed task 集合；使用
isolated time 推导 provisional release：

$$
\widehat r_h=\max_{i\in\mathcal B_h}\widehat f_i^{\mathrm{iso}},
$$

空 blocker 集合取 $\widehat r_h=0$。pooled tasks 按
$(-M_i,i)$ 排序，与 native runtime 的长任务优先队列一致。若 $A_h$ 是 group
$h$ 的 provisional available time，则依次执行：

$$
h_i=\arg\min_h(A_h,h),\qquad
\widehat s_i=A_{h_i},\qquad
A_{h_i}\leftarrow \widehat s_i+\widehat I_i(g).
$$

该 assignment 被降成一个可执行 DAG：每组的首个 pooled task 依赖
$\mathcal B_h$，后续 pooled task 依赖同组前一个 task。最后不是直接使用上述
isolated 和，而是把 DAG 交给现有 contention-aware event simulator：

$$
\widehat C_{\mathrm{tail}}(R,g)
=\operatorname{DAGCost}_{\widehat I,\widehat D}
  \left(\widehat{\mathcal G}_{R,g}\right).
$$

planner 在 strict 与全部合法 $(R,g)$ 候选间按相同 uncertainty band 和工作集
tie-break 选择。native runtime 只执行在线 group release 和原子领取；因此
planner 管 eligibility/width，runtime 管实际完成事件。surrogate assignment
可能因真实 contention 改变 group 完成顺序，仍需用 held-out E2E 数据校验，
不能把其预测值当作 exact dynamic makespan。

#### 2.3.1 单次 bounded tail repartition

production planner 可从一个 strict head DAG 派生受限尾部候选。令
$\operatorname{pred}(i)$ 和 $\operatorname{succ}(i)$ 分别为 task 的前驱和
后继，第二波 terminal 集合为

$$
\mathcal T=
\{i\mid \operatorname{pred}(i)\ne\varnothing,
          \operatorname{succ}(i)=\varnothing\}.
$$

当前实现仅在以下条件全部成立时扩展候选：

$$
|\mathcal T|=2,\qquad
\forall i\notin\mathcal T:\operatorname{pred}(i)=\varnothing.
$$

因此原 strict DAG 必须恰好由一个首波和两个第二波 terminal expert 构成，不允许
第三波，不允许改写仍有后继的任务。当前经过静态验证的默认域为 NUMA-local
96 cores，其有限宽度集合为

$$
\mathcal W_{\mathrm{tail}}(96)=\{24,32,48\}.
$$

其他 core domain 默认集合为空；只有调用方显式提供经过该域验证、满足
$w\mid C$ 且 $2w\le C$ 的有限集合时才扩展。候选宽度必须严格大于两个 tail
task 的原宽度。若经验表没有宽度 $w$，只允许在 serialized isolated formula
的线程域内，且 $M\ge36,\ M\bmod12=0$ 时插值；短 route、前两个 M12 panel 和
exact-M tail 仍要求真实 width calibration，缺失时直接剪掉该候选。

不切 M 时，两个新 fixed interval 为

$$
B_0=[0,w),\qquad B_1=[C/2,C/2+w),
$$

且 $2w\le C$。设首波 task $r$ 的原 interval 为 $S_r$，tail $j$ 的新前驱为

$$
\operatorname{pred}'(j)=\{r\mid S_r\cap B_j\ne\varnothing\}.
$$

profile 还可显式声明 route/M 切分数 $s>1$。当前 production 只考虑
$s\in\{1,2\}$；$s=2$ 必须同时满足

$$
M\bmod2=0,\qquad 4w=C,
$$

并且存在 exact-layout full-call anchor。两个 terminal expert 分别生成两个
连续 route slice：

$$
\mathcal R_{j,\ell}
=\left[\ell M/2,\;(\ell+1)M/2\right),\qquad
j\in\{0,1\},\ \ell\in\{0,1\}.
$$

采用 grouped placement：

$$
B_{j,\ell}=[(2j+\ell)w,\;(2j+\ell+1)w).
$$

因此 `M=1536,w=24,C=96` 时四个 task 分别是
`expert0[M0,M1]`、`expert1[M0,M1]`，占用
`[0,24),[24,48),[48,72),[72,96)`。每个 slice 的前驱仍由 interval overlap
确定：

$$
\operatorname{pred}'(j,\ell)
=\{r\mid S_r\cap B_{j,\ell}\ne\varnothing\}.
$$

planner 将首波 task 按 core 起点排在前面，再附加两个 tail task，因此所有新
前驱都指向更早 task。新 interval 互不重叠；每个 tail 只有在覆盖其新 interval
的全部首波 task 完成后才 ready。tail 在此之前尚未启动，所以这不是迁移、
抢占或运行中 resize，而是在 expert 边界预先选择另一 moldable width。runtime
继续执行普通 `strict` Plan V2，不做在线领取、不等待 preferred cohort。
每个 slice 独立 gather/pack、W13、W2，并写入互不相交的 route rows；只有同一
expert 的全部 slice 完成后才发布 expert completion，所以 ready-token merge
不会读取半完成的 expert。

候选只从 strict uncertainty band 和 point-estimate 最快的两个 head shape
派生，并由同一个 contention-aware DAG simulator 评分。working-set tie-break
使用首波与尾波瞬时工作集的较大者：

$$
S_{\mathrm{active}}
=\max\left(\sum_{r\notin\mathcal T}S_r^{\max},
           \sum_{j\in\mathcal T}\sum_{\ell=0}^{s-1}
             S_{j,\ell}^{\max}(w)\right).
$$

宽度相同但物理 interval 不同的 tail 不能共用 homogeneous shape derate。定义
placement-aware 校准签名

$$
q=(M,\ (t_r,b_r)_{r\notin\mathcal T},\
       (s,w,b_{j,\ell},\mathcal R_{j,\ell})_{j,\ell},\
       \text{kernel identity}),
$$

其中 $b$ 是 NUMA-local 物理 core 起点。若 profile 存在与 $q$ 完全一致的
bounded-tail full-call anchor $A(q)$，候选时间取

$$
\widehat T_{\mathrm{tail}}(q)=A_{\mathrm{median}}(q),\qquad
U(q)=\frac{\max(A_{\mathrm{median}}-A_{p10},
                A_{p90}-A_{\mathrm{median}})}
             {\sqrt{n_q}}.
$$

该 anchor 仅对 uniform route、相同 root shape/顺序、相同 tail 起点、宽度与
route-slice 数、相同 kernel/full-N geometry 生效，并且只做 exact-route
lookup，不插值或外推。$s=1$ 签名不匹配时回退 stage-aware DAG simulator；
$s>1$ 因包含同一 packed-B 被多个 team 同时扫描的额外 contention 状态，签名
不匹配时直接剪枝。因此 anchor 校准的是 interval placement 与 task-boundary
上下文，不修改 $I(M,t)$，也不会把一个 active-set 的 E2E winner 写入通用
isolated 曲线。

结果记录 `tail_repartition_width`、`tail_repartition_tasks`、
`tail_repartition_route_slices` 和候选数。默认 auto
同时比较 tail-pool 与 bounded tail；`dynamic_tail_pool=False` 且未显式指定
bounded 开关时保留旧 strict baseline。显式关闭 `bounded_tail_repartition`
时不生成候选；forced tail-pool 同样禁止同时派生该候选。cache key 额外绑定
bounded-policy identity；命中后重新运行轻量 LPT assignment 并校验 terminal
拓扑及 tail width 对新 route 的 isolated-model 支撑；任一不再合法时丢弃该
entry 并重新 cold search。

#### 2.3.2 实验性 strict 尾部任务领取

strict plan 的 lane dependency 是资源排序，不是 expert 间数据依赖。按
$\kappa=(t,\nu)$ 将 team 分成线程宽度 $t$、NUMA node $\nu$ 相同的 cohort，
记其 team 集合为 $G_\kappa$。对任意 $g\in G_\kappa$，planner 给出的有序
任务队列和固定 logical-core interval 为

$$
Q_g=(i_{g,1},\ldots,i_{g,n_g}),\qquad
B_g=[b_g,b_g+t).
$$

全部 $B_g$ 两两不交并恰好覆盖 logical workers；一个 team 内的物理 CPU 必须
属于同一 NUMA node，但不同 cohort 可以位于不同 node。迁移仅发生在同一个
$G_\kappa$ 内，因此不改变 task 的线程宽度，也不跨 NUMA。

给定很小的尾部深度 $d$，runtime 将每条 lane 分为 planner-owned 前缀和可迁移
后缀：

$$
d_g=\min(d,n_g-1),\qquad
P_g=(i_{g,1},\ldots,i_{g,n_g-d_g}),\qquad
U_g=(i_{g,n_g-d_g+1},\ldots,i_{g,n_g}).
$$

至少保留一个 task 在 $P_g$，因此前半程严格执行 planner 给出的 placement 和
lane 顺序；只有 $P_g$ 全部完成后，team 才进入后缀执行。自己的 $U_g$ 仍按原
team 顺序优先执行。令 task 状态
$x_i\in\{\mathrm{pending},\mathrm{running}(g),\mathrm{complete}\}$；
`running(g)` 在原子状态中编码实际 owner team，防止 donor worker 错误加入已被
迁移的 task。

当 team $g$ 的后缀已完成或已被其他 team 领取时，只有 leader 扫描同 cohort
的 peer 后缀。对 donor $h\in G_{\kappa(g)}$ 定义

$$
c_h=\left|\{i\in U_h\mid x_i=\mathrm{pending}\}\right|,\qquad
R_h=\sum_{i\in U_h,\ x_i=\mathrm{pending}}M_i,
$$

并选择

$$
h^*=\arg\max_{h\in G_{\kappa(g)},\ h\ne g,\ c_h\ge r_{\min}}R_h.
$$

leader 对 $U_{h^*}$ 中最早的 pending task 做一次
`pending -> running` CAS。CAS 成功后，整个 $t$-thread team 在自己的
$B_g$ 和 scratch 上执行该完整 expert。原 predecessor edge 只表示 donor
interval 的串行占用；迁移后实际 interval 与正在执行的 donor task 不相交，
所以不等待该资源边。expert 间真实数据依赖为空，TopK merge 仍由
expert-completion release 控制。CAS 失败立即重新扫描，不等待 donor，也不建立
跨 team barrier。leader 通过 per-team assignment epoch 唤醒其余 worker；
没有可领取 task 时立即释放 compute team，因为 pending 集合只会缩小，未来不会
产生新候选。若 ready-token drain 已启用，释放的 worker 继续扫描并执行自己的
固定 token owner 区间，直到所有 expert 和 token 发布完成。

当前 runtime 仅在以下条件全部成立时允许该动作：

1. strict Plan V2、fused SVE、fixed placement；
2. 每个 team 内为单 NUMA，全部 team 恰好分区覆盖 logical workers；迁移只在
   $(t,\nu)$ 相同的 cohort 内发生；
3. 每个 task 是唯一的 whole-expert task，不含 route slice；
4. dependency 为空或仅指向同 lane 的直接前驱，因此可证明是资源排序链；
5. scratch lease 前将每个 team 的 `max_rows` 扩到其 cohort 的最大 task route
   数，迁移期间不分配内存。

该动作不改变 $t_i$，因此 task 仍是非抢占 moldable job；变化的是未启动 task
的实际 placement 和 start time。它也不同于 `tail_pool`：planner 没有预先把
expert 标成 pooled，runtime 只在 planner 前缀结束后借用另一个 strict lane
的受限后缀。环境变量 `FUSED_CPP_MOE_STRICT_TAIL_STEAL=1` 显式启用；
`FUSED_CPP_MOE_STRICT_TAIL_STEAL_DEPTH` 默认 $d=2$，
`FUSED_CPP_MOE_STRICT_TAIL_STEAL_MIN_DONOR_TASKS` 默认
$r_{\min}=1$，即存在一个完整、未启动的合法后缀 task 就可领取。功能默认关闭，
planner candidate、cache identity 和 cost-model 评分均不变。现有 strict DAG
simulator 不解除 lane resource edge，因此不能
预测该实验路径；进入 production 前必须增加相同 suffix policy 的事件模拟和
held-out E2E 验证。

#### 2.3.3 已关闭的短 expert 重组与 residual-M suffix 候选

2026-08-12 对四类尾部候选做了同构 A/B。在线 `1/2/4/8T` cohort 重组、仅尝试
`2/4T` 的非阻塞重组、按实际完成时间触发的 residual-M 切分，以及 cost model
自动选择的 suffix DAG 均未改善 E2E；相关 runtime、schema 和 planner 实现已删除。
短任务队列的升序与现有 LPT 顺序差异也在噪声内，因此 ordering 可行域保持不变。

唯一稳定为正的几何是手工选定一个 terminal expert $i$ 和一个已经完成其末任务的
同宽 donor lane $h$，再通过现有 strict route-slice 合约把 $i$ 拆为两个连续 M 区间。
令原 lane 为 $g$、原 route 数为 $M_i$、线程宽度为 $t$，并定义

$$
q_i=\left\lceil\frac{M_i}{2}\right\rceil,
$$

则 Lab comparator 用两个不可抢占 fixed task 替换原 task：

$$
j_{i,0}=([0,q_i),g,t,\operatorname{pred}(g)),\qquad
j_{i,1}=([q_i,M_i),h,t,\operatorname{terminal}(h)).
$$

两片使用相同的正 `task_range_granularity=q_i`；当 $M_i$ 为奇数时第二片自然少一行。
两片全部完成后才发布 expert completion，因此 merge 可见性与未切分 expert 相同。
实验只接受 strict、尚未切片的 plan、同宽且互不重叠的完整 lane partition、terminal
target/donor 和 $M_i\ge24$；当前实测还把所有 CPU 限制在同一 NUMA node。

该静态几何在 DSV4 的手工 `E218/core40` 选择上重复提升 `0.48%--1.20%`，五次
均值约 `0.93%`，没有通过 2% production gate；其他四个 terminal target 在
`-0.19%--+0.05%`。因此它只保留为 Lab comparator，不加入 planner 候选、cache
identity 或 cost-model 评分。若未来重开，必须先给出能够区分 target/donor 的
held-out 选择规则，并把在线发现、交接和额外 task 开销显式计入目标函数。

#### 2.3.4 已退役的 W2 边界伸缩（历史模型）

本节仅保留实验知识，不属于当前可行域。active runtime、Plan V2 ABI、planner
bridge 和 benchmark CLI 已删除该路径；历史实现固定在 Git `0b58091`。通用 case
回退 `2.5%--24.8%`，唯一 `+3.66%` 的显式尾部迁移 case 已由 bounded tail
repartition 覆盖。以下公式用于解释历史实验和复现实测，不能作为当前 planner
候选。

elastic Plan V2 将完整 expert job 保持为同一个依赖节点，但在内部写成两个连续、
不可抢占的 stage：

$$
j_{i,13}=(i,\mathrm{W13},\bar t_i),\qquad
j_{i,2}=(i,\mathrm{W2},u_i),
\qquad u_i\in\{\bar t_i,p_i\}.
$$

第一版只允许 W2 宽度扩展：

$$
p_i>\bar t_i,\qquad p_i\bmod\bar t_i=0.
$$

记原 fixed interval 为

$$
S_i=[b_i,b_i+\bar t_i).
$$

planner 可令目标起点 $d_i=-1$，此时沿用包含原 team 的局部 cohort：

$$
B_i=
\left[
\left\lfloor\frac{b_i}{p_i}\right\rfloor p_i,\,
\left(\left\lfloor\frac{b_i}{p_i}\right\rfloor+1\right)p_i
\right).
$$

也可显式给出按 $p_i$ 对齐的 $d_i\ge0$：

$$
B_i=[d_i,d_i+p_i).
$$

合法目标满足

$$
S_i\subseteq B_i
\quad\lor\quad
S_i\cap B_i=\varnothing.
$$

因此只允许包含式扩容或完全不相交的 W2 team 迁移，禁止部分重叠。$S_i$ 和
$B_i$ 的全部 physical CPU 必须属于 task 声明的同一 NUMA node，禁止跨 NUMA
cohort。

令 $a_i=f_{i,13}$ 为 W13 完成事件，$\delta_i\ge0$ 为 planner 给定的有限
等待预算。runtime 在仍持有 $S_i$ 时，于 $a_i$ 尝试原子取得完整 $B_i$：

- $\delta_i=0$ 时，只把真正 idle 的额外线程视为可用；取得失败立即令
  $u_i=\bar t_i$；
- $\delta_i>0$ 时，同一 $B_i$ 内已经到达 W2-ready 的 base teams 可以组成
  cohort，按完整 W2 job 顺序使用 $p_i$ 个线程；
- 到达 $a_i+\delta_i$ 后，task 不再被新的 cohort 借用，并在已启动、不可抢占
  的有限 W2 job 释放原 team 后回退到 $\bar t_i$。

成功取得 $B_i$ 后，runtime 才释放 $S_i\setminus B_i$。这个
acquire-destination-before-release-source 顺序保证迁移不会产生无 owner
窗口；释放出的 source team 可在同一次 scheduler pass 中组成另一个 task 的
目标 cohort。

因此 timeout 约束的是“等待新 cohort 形成”的时间，不抢占已经开始的 W2。
若 task 在 deadline 前已把原 team 借给 cohort 中另一个 job，则其
W2-ready-to-assignment 延迟上界为

$$
W_i \le \delta_i + \max_{j\in B_i} T_{\mathrm{W2}}(M_j,p_i),
$$

且 deadline 后禁止再次借出，所以不会形成无限等待链。

该机制不抢占 GEMM，也不迁移或复制 packed-C intermediate；intermediate 位于
task-owned scratch，W2 的目标 team 通过 release/acquire 状态发布直接读取。
实际 $u_i$ 由真实完成事件决定，不是 cost model 的连续变量。第一版 planner
bridge 只显式生成 `2x8T->16T`、`4x2T->8T` 一类局部 cohort，或为指定 task
给出离散 W2 target；不把任意 stage width 或任意 cohort partition 加入
production 搜索空间。

#### 2.3.5 实验性全局两阶段计划

production Plan V2 的一个 task 始终是完整 expert，W13 完成后可立即进入该
expert 的 W2。为隔离这种 pipeline 与 vLLM 风格全局两阶段执行的差异，实验
entrypoint 也允许把每个 expert $i$ 表示为两个不可抢占 job：

$$
j_{i,13}=(i,\mathrm{W13}),\qquad
j_{i,2}=(i,\mathrm{W2}).
$$

W13 和 W2 分别生成完整 Plan V2，因而可以独立选择线程宽度、fixed interval 和
strict/tail-pool placement；两个 stage 都执行完整 N domain：

$$
(t_{i,13},q_{i,13},u_{13}(t_{i,13})),
\qquad
(t_{i,2},q_{i,2},u_2(t_{i,2})).
$$

两阶段之间存在全局 barrier。令

$$
B_{13}=\max_i f_{i,13},
$$

则

$$
s_{i,2}\ge B_{13}\quad\forall i.
$$

因此实验 makespan 为

$$
C_{\mathrm{2stage}}
=T_{\mathrm{route/setup}}
+C_{13}
+T_{\mathrm{barrier}}
+C_2
+T_{\mathrm{merge}},
$$

其中 $C_{13}$ 和 $C_2$ 分别由各自 stage DAG 的 contention event simulator
计算。`matched` 对照额外约束两个 stage 使用相同 plan；`independent` 候选
解除该约束。native executor 在进入 W2 前完整结束 W13 worker region，并为
每个 active expert 只 gather/pack 一次 A；它不采用旧 global N-range queue
的重复 A pack。该入口只用于 benchmark，不进入 production whole-expert
candidate space，也不改变默认 runtime。

定义活跃指示函数：

$$
a_i(\tau)=\mathbf 1[s_i\le\tau<f_i].
$$

令 $x_i(\tau)\in[0,1]$ 表示归一化完成进度。执行期间的真实速率为：

$$
\frac{dx_i(\tau)}{d\tau}
=\frac{a_i(\tau)}{I_i(t_i)D_i(\mathcal Z(\tau))},
\qquad x_i(s_i)=0.
$$

完成时间由下式隐式定义：

$$
f_i=\inf\{\tau\ge s_i\mid x_i(\tau)\ge1\},
$$

等价地：

$$
\int_{s_i}^{f_i}
\frac{d\tau}{I_i(t_i)D_i(\mathcal Z(\tau))}=1.
$$

当某个 expert 完成时，$\mathcal Z(\tau)$ 立即变化，剩余 expert 使用新的
slowdown 继续推进。若所有 $D_i\equiv1$，则退化为固定处理时间：

$$
f_i=s_i+I_i(t_i).
$$

### 2.4 硬约束与目标函数

核心原始问题只把 CPU 线程容量作为硬件可行性约束：

$$
\sum_{i\in\mathcal J}t_i a_i(\tau)\le C,
\qquad\forall\tau.
$$

LLC 容量和内存带宽在当前目标机器上是性能资源，而不是可执行性资源。工作集
超过某个 cache 比例时任务仍可运行，只是通过 $D_i(\mathcal Z)$ 变慢，因此不把
LLC 或总访存字节数写成原始问题的硬 cumulative constraint。若未来存在真正
不可违反的内存容量、affinity 或安全隔离限制，可额外扩展可行域。

调度域内的 expert compute makespan 为：

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
& t_i\in\{1,\ldots,C\}, &&\forall i,\\
& s_i\ge0, &&\forall i,\\
& \sum_i t_i a_i(\tau)\le C, &&\forall\tau,\\
& \int_{s_i}^{f_i}
\frac{d\tau}{I_i(t_i)D_i(\mathcal Z(\tau))}=1,
&&\forall i,\\
& C_{\max}=\max_i f_i. &&
\end{aligned}}
$$

在运行时比较不同 planner 时，还需要计入 planning latency：

$$
T_{\mathrm{runtime}}=T_{\mathrm{plan}}+C_{\max}.
$$

Dispatch、communication、scatter 和 combine 在不受 plan 影响时作为外部成本；
若其成本会随 plan 改变，则必须提升为目标函数的一部分。

### 2.5 可选的 ready-token combine 外层

当 combine 与 expert 尾部重叠时，token identity 会影响可行开始时间。令输入
token 集合为 $\mathcal Q$，token $q$ 的 TopK expert 前驱为：

$$
\mathcal P_q=\{e_{q,1},\ldots,e_{q,K}\}.
$$

完整 expert job 在 $f_i$ 时一次性发布其所有 route row，因此 token merge job
$q$ 的 release time 为：

$$
r_q=\max_{i\in\mathcal P_q}f_i.
$$

令 $u_q$、$g_q$ 分别为 merge 的开始和完成时间，$b_q(\tau)$ 为其一线程活跃
指示函数，则：

$$
u_q\ge r_q,
\qquad
\sum_i t_i a_i(\tau)+\sum_q b_q(\tau)\le C.
$$

runtime 将 token 连续静态映射到 logical worker。令
$L=\lceil|\mathcal Q|/C\rceil$，worker $c$ 的 owner 区间为

$$
\mathcal Q_c=[cL,\min(|\mathcal Q|,(c+1)L)),\qquad
o(q)=\left\lfloor q/L\right\rfloor.
$$

merge job $q$ 只能由 $o(q)$ 执行。owner 在 expert task 边界以及没有可运行
expert 时扫描自己的区间，仅执行满足 $r_q\le\tau$ 且尚未完成的 token。这一限制
保留 final merge 的连续 token locality，并消除全局 ready queue 的 claim 竞争，
代价是 owner 之间不做 merge work stealing。

令 $G_q=G(K,H)>0$ 为 token merge 的单线程 isolated time，$y_q$ 为其归一化
进度。对包含活跃 merge job 的扩展状态 $\mathcal Z^+(\tau)$，有：

$$
\frac{dy_q(\tau)}{d\tau}
=\frac{b_q(\tau)}{G_qD_q^{\mathrm{merge}}(\mathcal Z^+(\tau))},
\qquad
y_q(u_q)=0,\quad y_q(g_q)=1.
$$

若 merge 与剩余 expert 同时运行会改变 cache/带宽响应，还应把 expert slowdown
扩为 $D_i^+(\mathcal Z^+)$；不能直接假定被重叠的 merge 时间全部免费。包含
combine 的目标变为：

$$
C_{\max}^{+}=\max\left(\max_i f_i,\max_q g_q\right).
$$

当前 production planner 仍只优化 2.4 节的 expert-compute makespan，但 active
DAG simulator 同时返回每个 task 的预测完成时刻 $\widehat f_j$。route-sliced
expert 的预测完成时刻取其全部 slice 的最大值：

$$
\widehat f_e=\max_{j:e(j)=e}\widehat f_j,\qquad
\Delta_f=\max_e\widehat f_e-\min_e\widehat f_e.
$$

对 strict plan，若

$$
\Delta_f\le
\max\left(1\ \mathrm{ns},10^{-6}\max_e|\widehat f_e|\right),
$$

planner 写入 `early_merge=false`。此时模型预测所有 active expert 同时完成，没有
可供 combine 隐藏的 expert-compute 区间；executor 在 compute 后由全部 worker
按连续 token range 均匀分配 merge。

若 strict caller 同时提供本轮真实 `topk_ids`，planner 在选定 compute plan 后增加一个
不进入候选搜索的保守 gate。第一版只读取其 $(N,K)$ 形状，不扫描 token 内容；要求
`counts` 来自同一 `topk_ids`，且标准 TopK 保证一个 token 内 expert id 不重复。对 expert
$e$，令严格晚于其完成波次的 route occurrence 总量为

$$
R_{>e}=\sum_{j:\widehat f_j>\widehat f_e,\ \widehat f_j\not\simeq\widehat f_e}M_j.
$$

包含 $e$ 的 $M_e$ 个 token 中，最多 $\min(N,R_{>e})$ 个还能被更晚 expert 推迟。因此
$e$ 所在 ready 波次的 token 数满足可解释下界

$$
L_e=\max\left(0,M_e-\min(N,R_{>e})\right).
$$

当前 fixed-owner drain 的默认 batch 为 $B_{merge}=2$。若 $N\le B_{merge}T$，全部
token 本来就只占每个 owner 一轮，直接关闭 early merge。否则只需检查
$M_e\ge N-B_{merge}T$ 的高覆盖 expert；其余 expert 不可能单独证明目标 burst。令
$L^*=\max_eL_e$，若

$$
N-L^*\le B_{merge}T,
$$

则已有下界证明几乎全部 token 集中在一个 publication/drain 波次，波次外工作总量至多
为每个 owner 的一轮默认 drain，planner 写入 `early_merge=false`。令高覆盖集合为
$\mathcal H$；计算复杂度为 $O(E|\mathcal H|)$，且
$|\mathcal H|\le NK/(N-B_{merge}T)$，TP4 2048-token/96T 下最多约 6 个。该 gate 不增加
shape/order 候选，不改变 compute-plan histogram cache；每次调用只重新读取 $N$ 并用
本轮 task routes 计算下界。若不满足该充分条件、未提供 routing shape、模型不能返回
逐 task 时刻，或使用 stage-only/tail-pool plan，planner 写入 `null`，保留 runtime
heuristic。由于 $G_q$ 和 expert/merge contention 尚未进入目标，planner 仍不会强制
`early_merge=true`。

Plan V2 的 `early_merge` 是三态手动控制：`null` 为上述 auto，`true` 强制
ready-token 路径并跳过 team-load gate，`false` 强制统一 post-expert merge。
async ready-token executor 保持固定 expert core interval，
并在每个 expert task 完成的安全边界让各 worker 至多执行一个本 owner 区间内已经
release 的 token；没有可运行 expert 时使用同一规则继续扫描。compute 全部完成
后，同一轮 resident worker 分别排空自己的 owner 区间。尾部每轮至多处理
$B_{\mathrm{merge}}=2$ 个 token，并预取同批下一个 token。auto 默认只在
SVE direct-route 路径且
$\max_i\lceil M_i/8\rceil/t_i\ge1.25\min_i\lceil M_i/8\rceil/t_i$ 时生效。
`FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN=0` 恢复 early-ready 加连续 final merge；
`FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=0` 是覆盖 plan 的全局 kill switch；
batch/prefetch 环境变量继续控制尾部实现。

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

若多个 rank 共享内存带宽、cache 或互连，则 contention response 必须观察所有
rank 的联合活跃配置 $\mathcal Z_{\mathrm{global}}(\tau)$。一个 rank 完成后，
联合活跃配置随之改变，剩余 rank 必须切换到新的 slowdown，而不能将 dual-rank
速率应用到整个执行区间。

## 4. 问题类型与复杂度

该问题是 **带 active-set interference 的非抢占 moldable-job makespan
调度**。可以用非标准的描述性记号写成：

$$
P\mid\mathrm{moldable,\ nonpreemptive,\ interference}\mid C_{\max}.
$$

这里 `interference` 表示处理速率依赖同时活跃的 $(M_j,t_j)$ 配置，不是标准
Graham 三字段记号中的既有选项。当所有 $D_i\equiv1$ 时，问题退化为固定处理
时间的 moldable-job scheduling 特例。

### 4.1 混合离散-连续与非凸性

原始表达同时包含整数线程宽度 $t_i$ 和连续开始时间 $s_i$，因此更准确的说法
是 **混合离散-连续组合优化问题**，而不是纯离散问题。给定线程宽度和 job
相对顺序后，开始时间可以由有限个完成事件确定，所以困难部分主要来自有限但
指数规模的宽度、并发集合和顺序选择。

可行域不是凸集。严格地说，不是标量变量 $t_i$ “不是凸集”，而是它的离散
可行域

$$
\mathcal T_i=\{1,\ldots,T_{\max}\}
$$

不是凸集。凸集要求任取 $x,y\in\mathcal T_i$ 和 $\lambda\in[0,1]$，都有
$\lambda x+(1-\lambda)y\in\mathcal T_i$。当 $T_{\max}\ge2$ 时，取 $x=1$、
$y=2$ 和 $\lambda=1/2$，得到 $3/2\notin\mathcal T_i$，因此不满足该定义。
将线程宽度放松为连续区间 $[1,T_{\max}]$ 后，这一维可行域是凸集，但会引入
不可执行的分数线程数，所以它只是原问题的连续松弛。

即使固定所有线程宽度，非重叠调度仍然非凸。例如一颗核心上的两个单位时长
job 有两个可行调度：

$$
(s_1,s_2)=(0,1),\qquad (s_1,s_2)=(1,0).
$$

两者中点为：

$$
(s_1,s_2)=\left(\frac12,\frac12\right),
$$

此时两个 job 在 $[1/2,3/2)$ 重叠并违反单核容量约束。因此两个可行点的凸
组合可能不可行，可行域非凸。

### 4.2 最优解存在性

**命题 1（最优调度存在）**。设 $n=|\mathcal J|<\infty$、$C<\infty$，并且
$I_i(t)$ 与 $D_i(\mathcal Z)$ 满足第 2.2 节的有限性和 memoryless 假设。那么
存在一个可行调度 $S^*$，使：

$$
C_{\max}(S^*)
=\min_{S\in\mathcal F}C_{\max}(S).
$$

这里 $\mathcal F$ 是第 2 节定义的完整原始可行域，包含连续开始时间、任意合法
整数线程宽度和主动 idling。

**证明。**

首先构造一个有限的可行上界。令所有 expert 使用一个线程并任意串行执行。任意
时刻只有一个 expert 活跃，所以 $D_i=1$，该调度的 makespan 为：

$$
B=\sum_{i\in\mathcal J}I_i(1)<\infty.
$$

因此 $\mathcal F$ 非空，且最优值的下确界满足：

$$
\inf_{S\in\mathcal F}C_{\max}(S)\le B.
$$

任何 $C_{\max}>B$ 的调度都不会优于上述串行调度，所以只需考虑所有开始与完成
事件都位于 $[0,B]$ 的调度。

接下来固定线程宽度向量：

$$
\mathbf t=(t_1,\ldots,t_n)
\in\{1,\ldots,C\}^n.
$$

这种向量至多有 $C^n$ 个。每个 expert 有一个开始事件 $S_i$ 和一个完成事件
$F_i$。固定一个满足 $S_i$ 位于 $F_i$ 之前的合法事件顺序：

$$
\pi=(\pi_1,\ldots,\pi_{2n}).
$$

合法事件顺序数量有限，且不超过 $(2n)!$。令对应事件时间为：

$$
0=e_0\le e_1\le\cdots\le e_{2n}\le B.
$$

使用非严格不等式允许多个事件同时发生；任意 tied events 都可以按任意顺序
线性化，它们之间的零长度区间不产生执行进度。

对 $k=0,\ldots,2n-1$，令 $\mathcal A_k$ 表示事件顺序中前 $k$ 个事件发生后
仍然活跃的 expert 集合。区间 $[e_k,e_{k+1})$ 上的配置固定为：

$$
\mathcal Z_k
=\{(i,M_i,t_i)\mid i\in\mathcal A_k\}.
$$

该区间的 CPU 约束为：

$$
\sum_{i\in\mathcal A_k}t_i\le C.
$$

expert $i$ 的完成约束为：

$$
\sum_{k:\,i\in\mathcal A_k}
\frac{e_{k+1}-e_k}
{I_i(t_i)D_i(\mathcal Z_k)}
=1.
$$

固定 $\mathbf t$ 和 $\pi$ 后，$I_i(t_i)$、$D_i(\mathcal Z_k)$ 以及每个
$\mathcal A_k$ 都是常数。因此上述完成约束关于事件时间
$(e_1,\ldots,e_{2n})$ 是线性等式，事件顺序和时间范围是非严格线性不等式。
对应可行域 $P_{\mathbf t,\pi}$ 是 $[0,B]^{2n}$ 中闭集，因而闭且有界。根据
Heine--Borel 定理，$P_{\mathbf t,\pi}$ 是紧集。

在任意非空 $P_{\mathbf t,\pi}$ 上，最后一个事件必为某个完成事件，因此：

$$
C_{\max}=e_{2n}.
$$

这是事件时间的连续线性函数。根据 Weierstrass 极值定理，它在每个非空紧集
$P_{\mathbf t,\pi}$ 上都取得最小值。

最后，线程宽度向量和合法事件顺序都只有有限种。每个原始可行调度在
$C_{\max}\le B$ 时都对应至少一个 $(\mathbf t,\pi)$ 及其事件时间；同时发生的
事件由相等的 $e_k$ 表示。因此只需在有限个已取得的子问题最小值中再次取最小，
该最小值仍由某个具体调度 $S^*$ 取得。故：

$$
\exists S^*\in\mathcal F:\qquad
C_{\max}(S^*)
=\min_{S\in\mathcal F}C_{\max}(S).
\qquad\square
$$

该命题不是独立的 MoE 调度定理，而是 Heine--Borel 定理与 Weierstrass 极值定理
在当前有限事件模型上的推论。证明不要求 $I,D$ 具有解析公式；查表或 cost model
输出同样适用，只要它们对所有候选配置都有确定的有限正值。最优解存在不表示
能够在多项式时间内找到，也不表示 $(\widehat I,\widehat D)$ 上的最优解等于真实
$(I,D)$ 上的最优解。

### 4.3 固定 isolated-time 特例的判定版本

为证明困难度，只需限制到 $D_i\equiv1$。给定整数或经统一缩放后的有理数
$I_i(t)$ 以及 deadline $L$，定义：

$$
\textsc{Isolated-Moldable-Schedule}(L):
\quad
\text{是否存在可行 }\{s_i,t_i\}\text{ 使 }C_{\max}\le L\text{？}
$$

该判定版本属于 NP。证书包含每个 job 的线程宽度和开始时间。固定宽度后，
duration 为 $I_i(t_i)$；将所有开始和结束事件排序即可在多项式时间内验证 CPU
容量和 deadline。

对一般 active-set slowdown，是否能声称其判定版本属于 NP 取决于
$D_i(\mathcal Z)$ 的输入表示和求值复杂度。本文只用 $D_i\equiv1$ 的有限有理
特例证明强 NP-hardness，不对任意黑盒 slowdown 做更强的复杂度类别声明。

### 4.4 由 3-PARTITION 归约

给定一个 `3-PARTITION` 实例：$3m$ 个正整数 $a_1,\ldots,a_{3m}$ 和整数 $B$，
满足：

$$
\frac{B}{4}<a_i<\frac{B}{2},
\qquad
\sum_{i=1}^{3m}a_i=mB.
$$

构造 $E=3m$ 个 expert task，并令：

$$
C=m,\qquad D_i(\mathcal Z)\equiv1,
$$

$$
M_i=a_i,\qquad I(M_i,1)=M_i=a_i,
\qquad
I(M_i,t)=B+1\quad(2\le t\le m),
$$

deadline 为 $L=B$。由于任意 $t\ge2$ 都有 $I_i(t)>L$，所有 deadline 内的
可行调度都必须选择：

$$
t_i=1.
$$

问题因此退化为把 $3m$ 个处理时间为 $a_i$ 的非抢占 job 调度到 $m$ 个相同
核心上，并要求 makespan 不超过 $B$。总工作量正好是 $mB$，所以每个核心的
负载必须恰好为 $B$。又因为 $B/4<a_i<B/2$，每个核心恰好包含三个 job。

因此：

$$
C_{\max}\le B
\quad\Longleftrightarrow\quad
\{a_i\}\text{ 存在一个合法的 3-partition}.
$$

该构造为多项式归约，而 `3-PARTITION` 是强 NP-complete。因此：

- 固定 isolated-time 特例的判定版本是强 NP-complete；
- 原始 contention-aware makespan 优化问题是强 NP-hard；
- active-set slowdown、可变线程宽度和更丰富的执行响应不会消除该困难度。

## 5. Kernel Implementation Variant

Stage range 几何不是全局 implementation variant；它由 task 的 $(M,t)$ 在 plan
lowering 时决定。真正的全局 implementation variant（例如静态 asm 与 Xbyak
exact-M）$v\in\mathcal V$ 会诱导不同的真实响应：

$$
\Theta_v
=\{I_i^{(v)}(t),D_i^{(v)}(\mathcal Z)\}.
$$

对每个 variant 求解同一个调度问题：

$$
C^*(v)=\min_{s,t}C_{\max}(s,t;\Theta_v),
$$

再进行外层选择：

$$
C^*=\min_{v\in\mathcal V}C^*(v).
$$

因此 implementation variant 只改变真实性能环境，不改变问题类型。若在所有相关线程宽度和
活跃配置上，variant $v_1$ 都满足：

$$
I_i^{(v_1)}(t)D_i^{(v_1)}(\mathcal Z)
\le
I_i^{(v_2)}(t)D_i^{(v_2)}(\mathcal Z),
$$

则 $v_2$ 被 $v_1$ 支配，可以在进入 planner 前删除。Weight
split/range/window 不再是 implementation variant 或内部执行参数；给定 variant、
stage shape、N tile 和 team width 后，full-N owner geometry 唯一确定。

SVE 静态 asm 与 Xbyak exact-M 也属于不同的 $v$。两者虽然共享 packed weight、
N-split 和 M12 主体，但尾部映射不同，因此 profile identity 必须同时包含
`sve_implementation` 和 `m_tail_policy`。当前 production 默认是
`jit/xbyak_exact_m`；`asm/static_bucketed` 只作为 fallback、A/B 对照和历史 profile
解释，不允许用后者的 $\Theta_v$ 直接预测前者。catalog 的 `auto` 在进入
planner 前依次选择唯一匹配的 JIT、static calibration；不同 implementation
之间不能拼接 $\Theta_v$，同一完整 calibration domain 也不允许出现两份 profile。

实现也可以提供不暴露给 planner 的确定性 per-expert 子变体策略。令全局 backend
$b$ 固定，而 expert $i$ 的微内核/loop-order 子变体由只依赖本次调用已知 shape 的
映射决定：

$$
v_i=\pi_b(M_i,H,F,\mu),
$$

其中 $\mu$ 是机器能力与 build/runtime feature 集合。此时响应写为
$I_i^{(b,v_i)}(t)$ 和 $D_i^{(b,v_i)}(\mathcal Z)$。只要 $\pi_b$ 是确定性的、不会
改变线程宽度、依赖或资源可行性，它不扩展 planner 的决策空间，只把 backend
$b$ 的响应函数改成分段函数；profile identity 和验证仍必须记录该 policy 版本。

当前 x86 BF16 `auto` 是这种策略：runtime 支持时全局选择 AMX，否则回退
AVX-512 BF16；AMX 内再按每个 expert 的 route 数选择 M pattern 和 cache window。
环境变量只保留为强制实验/回归 override，不是 production 调用的必要输入。

## 6. 求解器编码

### 6.1 数学定义

核心定义直接使用整数变量 $t_i$。这保持问题表达简洁，不绑定具体求解器。

### 6.2 CP-SAT

当 $D_i\equiv1$ 时，CP-SAT 可以为每个 $(i,t)$ 建立 duration 为 $I_i(t)$ 的
optional interval，并要求每个 expert 恰好选择一个线程宽度；CPU 使用
cumulative constraint。此编码可作为 isolated fixed-duration 特例的离线
oracle，并返回可行上界和理论下界。

第一版离线 oracle 已实现于
`planners/isolated_cp_sat_oracle.py`。它接受显式线程宽度集合，删除满足
$t_1\le t_2$ 且 $I_i(t_1)\le I_i(t_2)$ 的被支配 mode，并以整数时间 tick
求解 whole-expert、固定宽度、非抢占 cumulative schedule。生产 planner 与
runtime 不依赖 OR-Tools；CLI 额外用相同量化后的 $I_i(t)$ 评估当前 strict
interval-DAG，输出同口径 regret。v1 不包含 dynamic tail pool、连续 core
interval placement、stage 内重划线程或 communication。

令 CP-SAT incumbent 为 $UB_s$，best objective bound 为 $LB_s$，则：

$$
LB_s\le C_{\mathrm{iso}}^*\le UB_s.
$$

对任意同一 mode 域中的可行 planner schedule $C_p$，即使求解超时仍有：

$$
\max\left(0,\frac{C_p}{UB_s}-1\right)
\le
\frac{C_p-C_{\mathrm{iso}}^*}{C_{\mathrm{iso}}^*}
\le
\frac{C_p}{LB_s}-1.
$$

当 $LB_s=UB_s$ 时得到 exact isolated regret。CLI 默认 1000 ns tick，以
$nq/2$ 报告 nearest-tick 的保守 critical-path 量化误差上界；需要最高模型精度
时可改为 1 ns。

在 `AmazonC5192Cores` TP4/F512 的 96-core rank 上，9 个默认 workload 的
第一轮验证表明该下界必须解释为 scheduling surrogate，而不是可达 wall time：
双 NUMA 当前实测相对 isolated optimum 区间高 50.4%-253.0%。其中
`5xM2040 + 174xM12` 的当前 `16T head + 1T tail_pool` 以 100 ns tick
达到 exact isolated optimum `6.1316 ms`，但双 rank wall time 为
`10.930 ms`；剩余差距因此不属于 isolated 线程宽度/开始时间选择。
完整数据、哈希和测量限制见
`optimizations/fused_moe_sve/results/amazon_192c_isolated_cp_sat_oracle_20260726.md`。

由于原始问题规定 $D_i(\mathcal Z)\ge1$，该 isolated 特例的最优 makespan
不大于真实 contention-aware 最优值，因此可以作为有效下界。

#### 6.2.1 Cold packed-B phase oracle

固定时长 oracle 允许任意数量的 cold-weight expert 同时保持 isolated rate，因而
对混合长短 expert 过于乐观。第二层离线 oracle 实现于
`planners/cold_phase_cp_sat_oracle.py`。对 expert $i$ 和 width $t$，令
$I_i(t)$ 为同一 empirical profile 的 isolated time，首个 SVE panel 行数为
$M_R=12$，则 cold reference duration 为

$$
c_i(t)=\min\left(I_i(t),I_{\min(M_i,M_R)}(t)\right),
$$

steady duration 为 $h_i(t)=I_i(t)-c_i(t)$。令 $W_i$ 为该 expert 的完整 W13/W2
packed-B 总字节数；默认 `expert` 粒度将 compulsory full-stage weight service
聚合成按顺序执行的两个 phase：

$$
P_{it}=\left[(c_i(t),W_i),(h_i(t),0)\right],
\qquad
b_i(t)=\frac{W_i}{c_i(t)}.
$$

第二个 phase 在 $h_i(t)=0$ 时省略。诊断用 `stage` 粒度展开完整 W13/W2 stage；
cold time 按 stage 字节比例分配，steady time 按原 stage phase duration 比例
分配，因此两种粒度都保持

$$
\sum_p d_{itp}=I_i(t),
\qquad
\sum_p W_{itp}=W_i.
$$

对每个 mode 建立 master interval $[s_{it},e_{it})$，其整个 lifetime（包括 phase
之间等待共享资源的时间）持续占用 $t$ 个 core。内部 phase 按序且不可抢占：

$$
s_{it,p+1}\ge e_{itp},
\qquad
e_{it}-s_{it}\ge\sum_p d_{itp}.
$$

令 cold phase 的平均 packed-B DRAM rate 为
$b_{itp}=W_{itp}/d_{itp}$，单 NUMA 可持续带宽为 $B_D$。首版增加两个
cumulative constraint：

$$
\sum_{i,t}t\,a_{it}(\tau)\le C,
\qquad
\sum_{i,t,p\in\mathrm{cold}}b_{itp}\,a_{itp}(\tau)\le B_D.
$$

CP-SAT 以整数时间和 bandwidth quantum 编码；若单个 cold phase 的量化 rate
超过 $B_D$，先把该 phase duration 下限提高到 $W_{itp}/B_D$。可选
`cold_phase_slots` 再限制同时 cold phase 数，但默认关闭。每个 expert 恰好选择
一个 width，master interval 不释放 team；因此 solver 可以错开 cold phase，不能
在等待时把其 core 借给另一个 expert。

同一模型同时求解两个域：fixed baseline 保留当前 lane DAG 且只允许 `8T`，但允许
solver 在 lane 内插入最优 cold-resource wait；mixed 域删除纯资源 lane edge，并允许
显式 width 集合。fixed incumbent 作为 mixed 的完整可行 warm start 和 objective
upper bound。若两者最优值分别满足

$$
F^*\in[F_L,F_U],\qquad O^*\in[O_L,O_U],
$$

则同一 surrogate 内 mixed 相对 fixed 的收益满足

$$
\max\left(0,\frac{F_L}{O_U}-1\right)
\le\frac{F^*}{O^*}-1\le
\max\left(0,\frac{F_U}{O_L}-1\right).
$$

该结果不是 contention-aware wall-time optimum 的严格证明：首版只限制 cold
packed-B，尚未限制 packed-A、store、LLC-to-L2 refill、频率/计算争用、merge 和
通信；`expert` 流体聚合还会把后续 W13/W2 cold range 前移。它的用途是回答
“允许混合 width 和主动 cold-phase 错峰时还有多少调度 headroom”，并为后续可执行
候选提供 surrogate 内的离线上限，不进入 production planner 或 plan cache identity。

runtime 反证进一步说明，不能把 $h_i(t)$ 的 lower-cache demand 直接置零。对 stage
$s$ 的一个实际 packed-B window，令 $\omega_{is}(t)$ 为 team 同时活跃的 packed-B
字节数，$p_i=\lceil M_i/M_R\rceil$ 为 A panel 数。若 team 按 N 切分，则每个 worker
需要保留的 B stripe 近似为 $\omega_{is}(t)/t$。用目标机器校准的私有 L2 retention
$r_2(x)\in[0,1]$ 表示大小为 $x$ 的 stripe 在相邻 A panel 间仍驻留 L2 的比例，则
完整 stage 的 lower-cache-to-L2 B 流量下界应写成

$$
Q^{B,L2}_{is}(t)=\omega_{is}(t)
\left[1+(p_i-1)\left(1-r_2\left(\frac{\omega_{is}(t)}{t}\right)\right)\right],
$$

而不是仅在首个 M12 phase 计一次 $\omega_{is}(t)$。再令时刻 $\tau$ 的共享 LLC
活跃窗口为

$$
S_s(\tau)=\sum_{i,t}\omega_{is}(t)a_{ist}(\tau),
$$

目标机器的 LLC retention 为 $r_3(S_s/C_{LLC})$，则 $Q^{B,L2}$ 中未由 LLC
保留的 replay 才继续形成额外 DRAM refill。$r_2,r_3$ 是较薄的硬件 service-curve
校准，不应由 route/thread 表逐点拟合。packed-A、gather、store 和 compute demand
仍作为独立资源推进。LLC 容量继续是 slowdown/candidate-pruning 因素，不把
$S_s\le C_{LLC}$ 错当成执行可行性的硬约束。

此外，静态 release $r_i$ 只是 task start 的 lower bound。若真实 duration
$\widehat I_i(t)>I_i(t)$，原先不重叠的区间会在 runtime 重叠，因此 oracle 时间轴上的
cumulative constraint 不再约束真实 active set。需要保持 cold stream 或 active
window 上界时，runtime 必须使用由 task completion 返还 token 的反馈 gate；单纯按
绝对时间 sleep/delay 不能维护该不变量。

一般 $D_i(\mathcal Z)$ 会使 job duration 随执行中的 active set 改变，不能直接
编码成一个固定 duration interval。要得到 contention-aware exact oracle，必须
进一步枚举并发 group mode、离散化时间/状态，或使用专门的 event-based search；
普通 CP-SAT interval 模型本身不是完整原始问题的等价编码。

#### 6.2.2 Cold-phase incumbent 的可执行 lowering

为验证 6.2.1 的 surrogate incumbent，实验 lowering 保持 oracle 给出的时间区间
$[s_i,e_i)$ 和线程宽度 $t_i$ 不变，再求每个 task 的连续逻辑 core 起点 $q_i$：

$$
0\le q_i\le C-t_i.
$$

时间重叠的两个 task 必须使用不相交的 core interval：

$$
[s_i,e_i)\cap[s_j,e_j)\ne\varnothing
\Longrightarrow
[q_i,q_i+t_i)\cap[q_j,q_j+t_j)=\varnothing.
$$

实现使用二维 no-overlap CP-SAT；求解后，每个物理 core 上相邻 task 可形成 dependency，
但 oracle 的开始时间只作为离线下界和 regret 解释数据，不再下沉到 Plan V2。曾实现的
per-task release gate 使 DSV4 mixed incumbent 吞吐回退 28.82%，且 0.25x release 与 eager
执行仅差 0.03%，说明错误来自 cold-only surrogate 对持续 cache/contention 的遗漏，而非
release 时钟精度。该 runtime ABI、主动 idling 与复现实验源码已退役；oracle 继续用于
计算不可争用上界和物理 placement，不声称其 duration 可直接执行复现。

### 6.3 MILP

标准 MILP 通常将 $t_i$ 展开为 one-hot 变量：

$$
y_{it}\in\{0,1\},\qquad
\sum_{t=1}^{T_{\max}}y_{it}=1.
$$

然后在线性 fixed-duration 特例中表示：

$$
t_i=\sum_t t\,y_{it},\qquad
I_i=\sum_t I_i(t)y_{it}.
$$

One-hot 是线程宽度的等价求解器编码，不是剪枝。一般 active-set slowdown 会
引入状态相关的非线性处理速率，需要额外变量或分段/枚举近似。

### 6.4 可证明资源下界：LB0 与 mode-relaxed LP

当前 planner 的最优性不能由某个已测 candidate 或 fixed-duration CP-SAT 单独证明。第一版
下界因此不构造可执行计划，只放宽任务的离散 mode、placement、开始时间和 contention，保留
任何真实执行都无法绕过的资源守恒与数据依赖。实现位于
`planners/resource_lower_bound.py`；SVE fused expert 的当前适配位于
`planners/sve_fused_expert_lower_bound.py`，不进入 production planner。该 adapter 的
$T^*$ 是“固定当前 exact-M SVE kernel、允许所列全部 width/mode”的调度最优值，不是允许
替换 GEMM 算法后的跨实现全局最优值。

令 phase 集合为 $\mathcal J$，phase $j$ 的合法 mode 集合为 $\mathcal M_j$，资源集合为
$\mathcal R$。mode $m$ 对资源 $r$ 的不可避免需求为 $q_{jmr}\ge0$，完整调度域的资源服务
上界为 $C_r>0$。归一化系数为：

$$
c_{jmr}=\frac{q_{jmr}}{C_r}.
$$

对每条不可重叠依赖链 $p\in\mathcal P$，把 mode-local wall-time 下界
$\ell_{jm}$ 当作该链对应的另一行系数：若 $j\in p$，则
$c_{jmp}=\ell_{jm}$，否则为零。以下统一用 $a\in\mathcal A=\mathcal R\cup\mathcal P$
表示资源行与链行。

**独立约束下界 LB0。** 每一行允许独立选择对自身最有利的 mode：

$$
LB_a=\sum_{j\in\mathcal J}\min_{m\in\mathcal M_j}c_{jma},
\qquad
LB_0=\max_{a\in\mathcal A}LB_a.
$$

不同 $a$ 的最小值可以来自互不兼容的 mode，因此 $LB_0$ 故意很松。对任意真实计划，phase
$j$ 最终采用的 mode $m_j$ 满足
$c_{jm_ja}\ge\min_m c_{jma}$；资源在 makespan $T$ 内最多提供 $C_rT$，依赖链各 phase
不能重叠，所以 $T\ge LB_a$ 对所有 $a$ 成立，进而：

$$
\boxed{LB_0\le T^*.}
$$

**mode-relaxed LP。** 为让所有资源行共享同一组 mode，引入凸组合
$x_{jm}\ge0,\sum_mx_{jm}=1$：

$$
\begin{aligned}
L_{LP}=\min\quad &T\\
\text{s.t.}\quad
&\sum_{j,m}c_{jma}x_{jm}\le T, &&\forall a\in\mathcal A,\\
&\sum_mx_{jm}=1, &&\forall j\in\mathcal J,\\
&x_{jm}\ge0.
\end{aligned}
$$

任意真实离散 mode 选择都映射为 one-hot $x$，再删除 placement、gang 同步、不可抢占、
contention 和离散性约束只会扩大可行域，因此：

$$
\boxed{LB_0\le L_{LP}\le T^*.}
$$

这里 LP 的 fractional mode 不是可执行计划，只是用于收紧下界。其对偶为：

$$
L_{LP}=\max_{\lambda\in\Delta_{|\mathcal A|}}
\sum_j\min_m\sum_a\lambda_a c_{jma},
\qquad
\Delta_d=\{\lambda\ge0:\sum_a\lambda_a=1\}.
$$

任何 simplex 中的 $\lambda$ 都给出合法下界。实现优先用可选 `oracle` extra 中的 GLOP
同时求 primal/dual；没有该依赖时使用 entropic mirror ascent。求解器浮点 objective 不作为
最终证书：输出的 $\lambda$ 投影到分母 $2^{40}$ 的整数 simplex，再用 Python `Fraction`
对原始输入系数重算 dual 并向下舍入；primal mixture 同样量化、精确重算并向上舍入。因此
即使 solver 未闭合，仍有：

$$
L_{dual}^{cert}\le L_{LP}\le U_{primal}^{cert},
$$

报告同时给出绝对/相对 gap、整数对偶权重和 fractional stage mixture，可独立复核。

**第一版 SVE lowering。** 对每个 active expert 生成
`W13 -> W2` 两个 phase，并为声明的线程宽度生成 mode。需求来自当前 exact-M mapper：

- physical/balanced BFMMLA work，包括 M1 的 M2 physical padding；
- key-body instruction、L1 load 与 fused epilogue work；
- 由 aggregate ceiling 和 busiest-lane per-core ceiling 推导的 $\ell_{jm}$；
- 对具有 per-core ceiling $C_r^{(1)}$ 的服务，以
  $q_{jm,core}=\max_r q_{jmr}/C_r^{(1)}$ 推导的总 core-time 下界；
- 可选的每个 active expert W13/W2 packed weight compulsory DRAM 首读。

这里不能用 $t\ell_{jm}$ 作为 core-time：当 N tile 不能被 $t$ 整除时，$\ell_{jm}$
可能由 busiest lane 决定，而其他 lane 可以提前结束。aggregate demand 除以 per-core
service ceiling 只要求所有 lane 合计必须消耗的 core-time，因此保持下界方向。

该 lowering 只保留所有合法 stage window 都必须执行的 inner-loop、epilogue 与首读工作；
省略 A/cache replay、range restart、gather、publication、merge 和 spill，因此不把 full-N
窗口错误限定为唯一可行 mode，只会让下界更松。`include_weight_dram` 默认关闭；只有 invocation
初态明确保证每份 active expert 权重至少从 DRAM 进入一次时才允许打开。

严格性还取决于容量口径：$C_r$ 必须是完整可行域上的**硬件服务上界**。架构最大发射率、
最大频率和内存通道理论上界可形成真实硬件证书；L1-hot GEMM、STREAM 或 PMU 实测峰值通常
只是已达到的下限，不能无条件作为真实容量上界。后者产生的是 calibrated analytic-model
下界，必须与 hardware-certified 下界分开标注。

当前未进入下界的项包括 stage-window 特有 replay、LLC/L2 容量 configuration、TopK ready
merge、跨 rank 通信和 NUMA placement；加入这些项前必须先证明其需求对声明初态和完整可行域
不可避免。调用方还必须给出声明调度域内完整的合法线程宽度集合；漏掉一个实际允许的 mode
可能抬高结果，此时证书只适用于显式受限域。该模块不改变第 7 节 production 搜索空间、剪枝
或 runtime。

**第一版验证。** 双资源、双 mode 的解析例中，两 mode 需求分别为 $(1,9)$ 与 $(9,1)$、
容量均为 1：实现得到 $LB_0=1$、$L_{LP}=5$，而枚举离散 mode 的最优值为 9。单测还覆盖
未收敛 mirror iterate 的 dual 有效性、critical chain、空 workload、非法 schema、JSON
certificate、M1 exact-M physical padding，以及 N tile 不均衡时不使用 $t\ell_{jm}$ 高估
core-time。

### 6.5 Production cold search 的 native 并行编码

schema-v2 empirical backend 的 cold miss 路径将剪枝后的 `IntervalPlanner`
搜索原样编码到 C++。Python 只负责解析 profile、生成合法 width/shape 集合并
导出不可变 calibration payload；C++ 负责 LPT assignment、task DAG 构造、
$T_{\mathrm{iso}}$、stage-aware event simulation、uncertainty band、tail-pool
surrogate 以及最终 tie-break。该迁移不改变 2.4 的目标函数、7 节的候选域或
候选排序键。

令 strict shape 数为 $S$，进入 dynamic 搜索的 head shape 数为 $H$，
workload 中去重后的短 route threshold 数为 $R$，pool width 数为 $G$，则待
评分候选数为：

$$
N_{\mathrm{cand}}=S+HRG.
$$

strict 候选彼此独立，先以固定 candidate index 并行评分。dynamic head 集合依赖
全部 strict makespan 和 uncertainty interval，因此 strict 阶段结束后存在一个
求解器级 join；之后 $(h,r,g)$ 候选再次独立并行。单个候选内部的 event
simulation 保持单线程，避免与候选级并行嵌套。若各候选计算量为 $W_j$、planner
worker 数为 $P$，则理想求值部分满足：

$$
T_{\mathrm{cold,eval}}
\gtrsim
\max\left(
\frac{\sum_j W_j}{P},
\max_j W_j
\right),
$$

另加 payload 构造、两阶段 join 和 Python bridge 生成成本。默认
$P=\min(8,C,P_{\mathrm{hw}})$，可由 `FUSED_CPP_MOE_PLANNER_THREADS` 或构造参数
覆盖；native 调用期间释放 Python GIL。worker 只写自己预分配的 candidate slot，
并按固定 index 收集结果，异常也按最小 candidate index 回到调用线程，因此
线程数不影响 plan、ranking 或 tie-break。schema-v1、analytic backend、旧扩展
和显式关闭 native 的情况继续使用 Python solver；cache hit 路径不重新执行
cold search。

## 7. 当前实现相对原始问题的剪枝

| 层级 | 原始可行域 | 当前限制 | 性质 |
| --- | --- | --- | --- |
| 线程宽度 | $1,2,\ldots,T_{\max}$ | empirical strict backend 为 `1,2,4,8,16,32`；analytic strict backend 使用 machine calibration 中显式允许的宽度；自动短 expert pool 只搜索 `1,2,4`；96-core bounded tail whole-expert 候选只搜索 `24,32,48`，其中缺表宽度仅允许长整 M12 formula 插值；route-sliced tail 只允许 exact-layout anchor 中显式校准的宽度，当前为 `24T`；forced override 可用其他已校准宽度；离线 cold-phase oracle 默认比较 `1,2,4,8,16`，不扩大 production 域。已拒绝的短 expert `1/2/4/8T` 在线重组完全从 active/runtime 搜索剪掉 | 离散宽度剪枝 |
| 并发配置 | 活跃 job 可形成任意满足 CPU 容量的 $(M_i,t_i)$ 组合 | 搜索静态 core shape，并自动比较 strict、threshold/统一宽度 tail-pool 与恰好两个 terminal expert 的一次 bounded repartition；后者可在 exact anchor 命中时把每个 terminal expert 切成两个连续 M slice，使四个 fixed task 覆盖全部核心。benchmark-only large/small 候选按一个 route threshold 把同 NUMA 核心切成两个互不重叠区域，从调用开始并发执行两类 whole expert；已反证的三类 bounded-stream 扩展再保留固定数量的 M<=12 1T 区，只用于实验 | static-partition + terminal repartition 剪枝 |
| Shape 集合 | 所有满足 CPU 容量的整数宽度组合 | empirical backend 只用 profile shape；analytic backend 生成 homogeneous 和至多两种宽度的 shape，再应用 active 工作集规则；tail-pool 和 bounded tail 只从 strict uncertainty band 和最快两个 head shape 派生。large/small 与三类 bounded-stream 实验只允许 profile 已校准宽度、连续且可整除的 core region，不进入 production 搜索 | 候选剪枝 |
| Assignment | 任意 expert-to-resource 调度 | 先按 isolated cost 的 LPT 固定 expert-to-lane membership；非 full-call-anchor strict 候选再比较原顺序与两个奇偶 lane 反序 seed，但不把 expert 移到另一 lane。large/small 实验在两个区域内分别做 LPT；三类 bounded-stream 在 large/medium/short 区内分别做 LPT，均禁止跨区迁移。实验 strict tail-steal 保留每条 lane 的 planner 前缀，只从同 $(threads,NUMA)$ cohort 的 peer lane 受限 pending 后缀迁移 whole expert | 启发式分配与 suffix-steal 剪枝 |
| Runtime plan contract | task 可携带离散宽度集合、stage、route-slice、resize 边界、动态 placement 和合法的 tile-aligned owner window | bounded tail 在 terminal expert 启动前生成新的 singleton fixed width 和 blocker DAG；production whole-expert task 的 W13/W2 均执行完整 N domain，Plan/ABI 不携带 weight range、split、byte-window、task release 或 W2 resize，但可逐 stage 携带 per-worker `window_tiles`（0 表示 full stripe）；窗口只改变完整 N domain 的访问顺序。production 由确定性 band policy 在 width 选定后给值，analytic v6 仅作 shadow post-policy，不把 window 扩成 planner 自由变量；exact-anchor route fission 将同一 expert 的连续 M slice 作为多个 strict task，只有全部 slice 完成后才发布 expert completion；tail_pool 保持 whole-expert 动态 placement；实验 strict tail-steal 只接受 fixed、whole-expert 资源链和恰好覆盖 worker 的不重叠 team，并仅在同线程宽度、同 NUMA cohort 内迁移，同时保留 ready-token drain。手工 residual-M suffix 只复用现有 strict route-slice 表示作为 Lab comparator；不新增 resize、placement 或 schema 语义 | 单次 terminal expert 重分区、结构化 window 候选与受限未启动 task 迁移剪枝 |
| Stage coupling | W13/W2 可形成任意满足依赖和容量的 stage DAG | production 使用 whole-expert pipeline；独立 W13/W2 Plan V2 加全局 barrier 仅作为实验 entrypoint，matched/independent 两种计划都不进入默认搜索 | production 粒度剪枝与实验对照 |
| x86 synchronous executor team mapping | expert 可取任意合法整数宽度并形成任意 wave | API 接受 1--256 workers；均衡 route 用 atomic expert queue，active expert 不足时按 route/当前宽度贪心组 team，强偏斜时按 64-row target 形成有序 wave | planner 外的确定性 runtime mapper |
| Ordering | 任意可行开始时间和顺序 | strict 候选只比较三种确定性顺序：全部 lane 保持 LPT、奇数 lane 整链反序、偶数 lane 整链反序，并用完整 event-time cost model 严格判优；不搜索任意排列或主动 start delay。full-call anchor 保持原 LPT。实验 strict tail-steal 保留 planner 前缀和本地后缀优先，只允许领取 peer lane 的 pending suffix frontier | 顺序剪枝 |
| Idling | 允许主动等待以避开争用 | planner 可关闭 tail-pool 和 bounded tail 保留原 strict；bounded tail 只依赖 blocker 完成、不增加主动等待；tail-pool 保持 non-idling；strict tail-steal 找不到满足 $c_h\ge r_{\min}$ 的后缀后立即释放 compute team，并在启用时转入 ready-token drain；ready-token 路径仅填充无可运行 expert 的空闲 lane；已拒绝的 cold-phase task-release、W2 cohort timeout 和短 expert cohort barrier 不再属于 runtime 可行域 | 仅保留 non-idling 与依赖边界剪枝 |
| Workload 输入 | 任意合法 global 或 rank-local route histogram | planner 接受任意 histogram；catalog preset 只扩展验证覆盖，不过滤运行时输入 | 不剪枝 |
| Route combine | 任意满足 TopK release 约束和 CPU 容量的 merge 排程 | planner 不搜索 combine service time；strict plan 在预测 expert 同时完成时关闭 early merge；若 caller 提供真实 TopK shape，选定 compute plan 后用 route counts 减去更晚波次 route 总量，得到单 expert 所在 ready burst 的保守下界；当该下界证明 burst 外 token 不超过一轮默认 owner drain（$2T$）时同样关闭。其他情况保留 auto，且从不由该 gate 强制开启；runtime 将连续 token range 固定映射给 logical worker，在 expert 边界和空闲期处理本 owner 已 release token，并在同一 resident worker job 排空；owner 间不偷取 merge；Plan V2 允许显式 on/off | 不扩大搜索空间的 routing-shape/route-bound 保守后处理 |
| Kernel variant | 任意未被支配的实现 | ARM empirical identity 只包含机器/拓扑、分布式 expert shape、SVE implementation/tail policy、`full_n_team_stripes` 和 source/binary hash；每个 identity 只允许一个活动校准，旧 split/range profile 不兼容。`auto` 优先选择唯一匹配的 `jit/xbyak_exact_m` 校准，否则回退唯一匹配的 `asm/static_bucketed` 校准。每个 shape/tail-pool 候选确定 team width 后，empirical/analytic model 直接按 $u_s(t)$ 计算 owner stripe，不生成额外执行参数。x86 AMX 使用不进入 planner 的确定性 per-expert pattern/cache policy，AVX-512/AMX 共用确定性 team-N/wave policy | 实现候选限制与 width-derived runtime geometry |
| Isolated time | 真实 $I_i(t)$ | production 默认仍为经验公式；可选 analytic backend 由 kernel demand、cache traffic 和机器 service curves 计算 | cost 近似，不剪枝可行域 |
| Contention | 任意动态活跃配置上的真实 $D_i(\mathcal Z)$ | production 默认为实测 profile；bounded tail 仅在 uniform route、root/tail width 与物理 interval 完全匹配时使用 exact-layout full-call anchor，且禁止 route 插值；未命中仍走 stage-aware simulator；实验 strict tail-steal 暂不进入 cost model；analytic backend 按 L1-hot M12 GEMM core、L2/LLC/DRAM/epilogue 共享容量推进事件，register-only matrix/frontend/L1 只保留诊断。machine schema-v2 可按显式 CPU placement 把 LLC 服务/容量分解到 LLC 域并受 rank fabric cap 限制，DRAM 始终为 NUMA-rank 共享；当前 planner DAG 尚不携带该 placement，故 production scoring 仍走 rank 聚合兼容路径。cold-phase oracle 只约束首个 M12 packed-B DRAM phase，运行时验证已证明它不能替代 per-worker L2 retention、完整 active full-stage working set、LLC-to-L2 service、容量和 active-set slowdown。large/small 实验在 cross-class service 未过绝对时间 gate 前只按实测比较，不接受 isolated-LPT 或当前全局 derate 的生产评分；三类 bounded-stream 已证明 isolated short-byte 配额低估所需并发，但严格配对实验未发现 grouped/crossed M/S 顺序有显著差异 | cost 近似，不剪枝可行域 |
| 跨 rank lifetime | 每个 rank 的资源状态随其他 rank 完成而变化 | 有 matching single-rank companion 时，多 rank 活跃阶段使用 concurrent-rank profile，最后一个 rank 的剩余 phase 切换到 single-rank profile；缺表时保守保持 concurrent-rank rate | cost 状态近似，不剪枝可行域 |

当前 `IntervalPlanner` 搜索的是上述剪枝后 plan space 中的方案，不是原始问题
的全局最优方案。

Amazon 192-core NUMA0 的 schema-v2 empirical profile 已将实测线程域扩展到
`1,2,4,8,16,32,48,64,96`，但这只扩大 $\widehat I_i(t)$ 的校准域。当前 planner 的
empirical 线程宽度剪枝仍为 `1,2,4,8,16,32`；在完成 48/64/96T 的 held-out
regret 验证前，不自动扩大该 backend 的在线决策空间。analytic backend 不继承
此实测表限制，但 machine calibration 必须显式列出可执行宽度。

2026-07-26 的双 NUMA TP4/F512 配对刷新同样不扩大该在线决策空间。当前二进制
下 96T isolated 的 split/no-split 中位时间相对旧表分别增加 73.25%/78.74%，
且两个 NUMA 的中位 max/min 差为 24.62%/31.25%；这些点继续作为测量边界和
不确定性输入，不能据此解除线程宽度剪枝。

## 8. 当前 Cost Model 的位置

原始问题中的 $I_i(t)$ 和 $D_i(\mathcal Z)$ 是目标机器上的真实但未知响应。
Cost model 分别提供估计：

$$
\widehat I_i(t)\approx I_i(t),
\qquad
\widehat D_i(\mathcal Z)\approx D_i(\mathcal Z).
$$

当前 stage-aware event simulator 使用这两个估计推进 job：

$$
\frac{dx_i(\tau)}{d\tau}
=
\frac{a_i(\tau)}
{\widehat I_i(t_i)\widehat D_i(\mathcal Z(\tau))},
\qquad
x_i(s_i)=0,\quad x_i(f_i)=1.
$$

因此 cost model 只近似目标函数中的真实性能响应，不改变由 route tasks 和 CPU
容量定义的原始可行域。

2.5 节的 ready-token merge service time 尚未进入 active cost model。当前
profile 的 $\widehat I_i,\widehat D_i$ 仍只描述 expert compute；新增逐 task
$\widehat f_j$ 与 `early_merge=false` 只识别“预测无重叠窗口”的情况，不估计
combine 收益。在增加 merge service time、expert/merge 异构 contention 和真实
分布留出验证前，planner 不得把实验重叠时间当作确定收益。

### 8.1 当前 active isolated model

新生成且携带 `iso_formula` 的 schema-v2 profile 使用：

$$
\widehat I_i(t)
=T_{\mathrm{iso}}(M_i,t)
=O(t)+C(M_i)\phi_{\mathrm{USL}}(t)k_\phi(t),
$$

$$
O(t)=o_0+\frac{o_1}{t},\qquad
\phi_{\mathrm{USL}}(t)=\frac{1+\alpha(t-1)+\beta t(t-1)}{t}.
$$

$C(M)$ 是单线程 route-work 的一维实测校准曲线；$O(t)$ 和
$\phi_{\mathrm{USL}}(t)$ 分别表示固定开销与相对理想 $1/t$ 的线程效率基线，
$k_\phi(t)$ 是与 route 无关的一维实测校正。Xbyak 路径的 exact M1--M12 与前两个
完整 M12 panel 的启动/线程效率不同于 steady-state M12 bulk，因此保留每个
`(tail, threads)` 的实测 residual；更大的 M 使用公式，并按 kernel 组合规则
追加 exact remainder。历史 static-asm profile 继续按 M1/M2/M4/M8 bucket 解释，
两种 residual 不可混用。公式仅在 profile 的已校准线程域内有效，不承担跨机器
或更高线程数外推。没有序列化公式的历史 profile 继续使用旧二维 table，避免
离线重拟合改变既有 planner 基准；可显式指定 `iso_mode="formula"` 做 shadow 对比。

Amazon 192-core NUMA0 的 TP4/TP2/EP4/EP2 split/no-split profile 继续使用同一
公式，`thread_domain` 扩展到 `[1,96]`，并以 48/64/96T 实测点校正
$k_\phi(t)$；公式形式及小 M residual 组合规则没有改变。

2026-07-26 的双 NUMA TP4/F512 资源刷新只重新拟合参数和一维
`C(M)`/`k_phi(t)` 校正，不改变上述方程。对新表 M>=48 的实测点，split/no-split
公式绝对误差中位数为 1.91%/1.00%，P90 为 9.66%/9.11%；36.17%/42.66% 的最大
误差集中在宽 team。no-split 基础 USL 拟合得到负 $\alpha$，但序列化
`phi_pts` 仍直接校正已测线程域；因此 $\alpha,\beta$ 只作为分离拟合参数，
不得解释为跨机器物理常数或外推到 96T 之外。

#### 8.1.1 全局两阶段的首版 isolated 分解

现有 empirical profile 只直接校准完整 fused expert 的
$\widehat I_i(t)$，尚无同网格的独立 W13/W2 isolated 表。实验
`PlannedTwoStagePlanner` 首版先将

$$
\widehat I_i(t)=O_i(t)+C_i(t)
$$

分解为：

$$
\widehat I_{i,13}(t)
=O_i(t)+\frac{2}{3}C_i(t),
\qquad
\widehat I_{i,2}(t)
=\frac{1}{3}C_i(t).
$$

$2/3$ 与 $1/3$ 来自 TP4/F512 形状下 W13:W2 的 useful GEMM FLOP 比；
route/setup、gather 和 pack-A 固定项全部计入 W13，使 combined stage plan
只收取一次 expert 启动开销。每个 stage 再按其实际 packed-B range 数和
working-set geometry 拆分 phase，并用原 contention model 推进。operator
call setup 只在两个 stage 合计时收取一次。

该分解是为了给独立 stage shape 搜索提供可执行的 first-order surrogate，
不是绝对时间结论。它忽略 SiLU/packC 与 direct-route store 的不同比例，也
未从完整 expert 样本中辨识 stage-specific residual。要把 global two-stage
候选加入 production，必须先采集同一 kernel、route、thread、window 网格下的
独立 W13/W2 timing，并在未参与拟合的 mixed distribution 上验证 stage
ranking 和 E2E regret。

### 8.2 实现弱相关的分层 GEMM cost model

简单的 $\max(F/P,Q/B)$ 无法表示当前 BF16 microkernel 的 L1 重复加载、
矩阵指令发射、不同 cache 层、tail padding 和 fused epilogue；把 M12、BFMMLA
等细节直接写入核心公式又会使 planner 绑定单个实现。因此 shadow model 使用
算法工作量、kernel 映射和机器响应三层契约。

对 stage $s$，设问题实例为 $x$，调度决策为 $\sigma_s$，kernel implementation
为 $\kappa$，目标机器为 $\mu$。完整分解为：

$$
w_s=\mathcal A_s(x),
$$

$$
d_s=\Phi_\kappa(w_s,\sigma_s),
$$

$$
\widehat T_s
=\Psi_{\mu,\kappa}(d_s,\sigma_s)
+O_{\mu,\kappa}(\sigma_s).
$$

$\mathcal A_s$ 只产生逻辑工作 `LogicalGemmWork`；$\Phi_\kappa$ 是可替换的
kernel mapper，产生统一 `KernelDemand`；$\Psi_{\mu,\kappa}$ 使用绑定
`machine_id + implementation_id + active_threads` 的实测
`MeasuredMachineProfile`。stage 预测组合成 $\widehat I_i(t)$，并可作为
$\widehat D_i(\mathcal Z)$ 的实现特征；物理 demand 留在 cost-model/profile
内部或用于候选剪枝。planner 的原始接口不需要知道 M12、SVE 或 BFMMLA。

#### 8.2.1 算法工作量

对 route 数 $M$、hidden size $H$ 和 intermediate size $F$，逻辑 GEMM 为：

$$
\mathrm{W13}:[M,H][H,2F],
\qquad F_{13}^{\mathrm{use}}=4MHF,
$$

$$
\mathrm{W2}:[M,F][F,H],
\qquad F_2^{\mathrm{use}}=2MHF.
$$

当前 BF16 输入/权重、BF16 fused intermediate、每元素 $b_{\mathrm{down}}$ bytes
的 down route store，其 compulsory one-pass bytes 为：

$$
Q_{13}^{\min}=2MH+4HF+2MF,
$$

$$
Q_2^{\min}=2MF+2FH+b_{\mathrm{down}}MH.
$$

production direct-route 默认 $b_{\mathrm{down}}=2$；历史 FP32 down buffer 为
$b_{\mathrm{down}}=4$。该值是 kernel/output policy，不改变 useful FLOPs。

给定机器理论上界 $P_\mu^{\mathrm{peak}}$ 和
$B_\mu^{\mathrm{peak}}$，算法层可以给出条件下界：

$$
T_s^{\mathrm{LB}}
=\max\left(
\frac{F_s^{\mathrm{use}}}{P_\mu^{\mathrm{peak}}},
\frac{Q_s^{\min}}{B_\mu^{\mathrm{peak}}}
\right).
$$

这些是算法可解释的有用 FLOPs 和至少读写一次所有逻辑 tensor 的下界，不是
实际 cache 流量；$T_s^{\mathrm{LB}}$ 也只是理论下界，不直接作为运行时间预测。
算法层不包含线程数、N tile、M panel、padding 或 split；因此不满足某个 kernel
对齐要求的逻辑 shape 仍是合法问题实例，只可能在特定 mapper lowering 时被拒绝。

#### 8.2.2 Kernel 执行映射

统一的实现需求向量写作：

$$
d_s=\left(
F_s^{\mathrm{exec}},I_s,
Q_{L1,s},Q_{\mathrm{private},s},Q_{\mathrm{shared},s},
E_s,W_s,N_{\mathrm{call},s},N_{\mathrm{range},s}
\right).
$$

其中包含实际执行 FLOPs、关键指令、各层流量、epilogue 元素、瞬时工作集、
调用/range 次数和 busiest-thread balanced work。M12 换成其他 tile、SVE 换成
NEON 或改变 tail 规则时，只替换 $\Phi_\kappa$，不修改算法公式或 planner。

当前 `SveBf16KernelProfile` 对应 implementation
`sve_bf16_xbyak_exact_m1_m12_v2`。令 $\mathcal P(M)$ 为物理 panel 序列，每个 panel
$p$ 分别记录逻辑行 $m_l(p)$、实际计算行 $m_c(p)$、packed-A 行 $m_p(p)$ 和
store 行 $m_s(p)$。写 $M=12b+r,0\le r<12$，前 $b$ 个 panel 均为
$(12,12,12,12)$；若 $r>0$，exact tail 为：

$$
(m_l,m_c,m_p,m_s)
=\left(r,2\left\lceil\frac r2\right\rceil,
\begin{cases}8,&r\le8\\12,&r>8\end{cases},r\right).
$$

例如 M1 为 $(1,2,8,1)$，M5 为 $(5,6,8,5)$，M9 为 $(9,10,12,9)$。
static-asm control profile 则仍将 3--4、5--8、9--11 分别映射到
M4、M8、M12 compute/store bucket。设 SVE vector 为 $V$ bytes，BF16 N tile
$\nu=V/2$，tile 数 $q=N/\nu$，则当前 JIT mapper 产生：

$$
I_{\mathrm{BFMMLA}}(p)=\frac{m_c(p)Kq}{2},
\qquad
I_A(p)=\frac{m_c(p)Kq}{8},
\qquad
I_B(p)=Kq,
$$

$$
Q_{L1,A}=16\sum_p I_A(p),
\qquad
Q_{L1,B}=V\sum_p I_B(p).
$$

若顺序 N range $j$ 有 $q_j$ 个 tile，当前实现的保守 shared-cache 映射为：

$$
Q_{\mathrm{shared},A}
=2K\sum_p m_c(p)\sum_j\min(t,q_j),
$$

$$
Q_{\mathrm{shared},B}=2KNP,
\qquad P=|\mathcal P(M)|.
$$

split/no-split 是传给 mapper 的通用 $\sigma_s$，不是算法语义或核心公式中的特殊
分支。当前 SVE mapper 中，split 不改变 BFMMLA、L1 A/B、aggregate B 或 C，
但改变 range 次数、瞬时 weight 工作集，并在保守口径下增加 shared-A scan。
Exact-M 相对 static bucket 不改变每个 panel 的 aggregate B load；它只减少被
padding 的 row pair 对应的 BFMMLA、A broadcast 和无效 epilogue/store 工作。

#### 8.2.3 实测机器响应

当前 ECM response 使用：

$$
T_{\mathrm{body}}
=\max\left(
\frac{F_s^{\mathrm{bal}}}{P_{\mu,\kappa}(t)},
\frac{I_s^{\mathrm{bal}}}{R_{\mu,\kappa}(t)},
\frac{Q_{L1,s}^{\mathrm{bal}}}{B_{L1,\mu,\kappa}(t)}
+\frac{Q_{\mathrm{private},s}}{B_{\mathrm{private},\mu,\kappa}(t)}
+\frac{Q_{\mathrm{shared},s}}{B_{\mathrm{shared},\mu,\kappa}(t)}
\right),
$$

$$
\widehat T_s
=O_{\mathrm{stage}}+N_{\mathrm{range},s}O_{\mathrm{range}}
+T_{\mathrm{body}}
+\frac{E_s^{\mathrm{bal}}}{R_{\mathrm{epi},\mu,\kappa}(t)}.
$$

所有 service rate 和固定开销必须由目标机器上该 implementation/thread width 的
独立 microbenchmark、PMU 或明确的 stage pair 测量得到。只观察一个 stage
latency 只能得到 $F/\Delta T$、$Q/\Delta T$ 等同一时间的等价 required rate，
不能同时识别计算、L1 和 shared-cache ceiling。

`gemm_cost_model.py` 实现通用契约和机器响应，`sve_bf16_kernel_model.py` 实现
当前 SVE mapper，`gemm_ecm.py` 保留原 CLI/API facade 与诊断报告。该路径仍是
shadow，不替换 active $T_{\mathrm{iso}}$。V3 steady-M12 留出误差低于 1.7%，
但只验证同 shape 的 panel 线性，尚未证明跨 shape/kernel/machine 的可迁移性；
完整限制见 `cost_model/GEMM_ECM_VALIDATION.md`。

#### 8.2.4 解析机器响应与共享资源模型

`analytic_model.py` 将 8.2 的 demand 契约提升为可被 `IntervalPlanner` 直接使用
的可选 backend。它不加载 route/thread latency table 或 contention shape。对
每类资源 $r$，机器校准只保存单核速率 $R_{r,1}$、饱和聚合速率
$R_{r,\mathrm{sat}}$ 和饱和线程数 $t_{r,\mathrm{sat}}$。matrix/frequency
derate 可选 `power`：

$$
\alpha_r
=\frac{\log(R_{r,\mathrm{sat}}/R_{r,1})}
{\log t_{r,\mathrm{sat}}},
\qquad
R_r(t)
=\min\left(R_{r,\mathrm{sat}},R_{r,1}t^{\alpha_r}\right).
$$

初段近线性、之后撞到共享 fabric ceiling 的 cache/DRAM 可选
`shared_bottleneck`：

$$
\beta_r
=\frac{R_{r,1}t_{r,\mathrm{sat}}/R_{r,\mathrm{sat}}-1}
{t_{r,\mathrm{sat}}-1},
\qquad
R_r(t)
=\min\left(
R_{r,\mathrm{sat}},
\frac{R_{r,1}t}{1+\beta_r(t-1)}
\right).
$$

两种曲线使用相同三个硬件锚点；curve family 表达资源拓扑，不增加
route-dependent 参数。

schema-v2 machine calibration 对 LLC/DRAM 不再强行压缩为三个锚点。设独立
service probe 在严格递增线程点 $t_1=1<t_2<\cdots<t_k$ 得到聚合速率
$\widetilde R_j$。先用等权 isotonic regression 得到单调速率 $R_j$，再在线段内
插值：

$$
R(t)=R_j+\frac{t-t_j}{t_{j+1}-t_j}(R_{j+1}-R_j),
\qquad t_j\le t\le t_{j+1}.
$$

$t\ge t_k$ 时取 $R_k$。采样点内误差为零或仅包含 isotonic 去噪残差，这是插值
性质，不能当作模型精度；外推/选择精度必须用 held-out 线程宽度报告。

对一个 NUMA rank 内的 LLC 域集合 $\mathcal D$，schema-v2 显式保存每个域的
CPU 集 $\mathcal C_d$、容量 $C_d$ 和域内 refill 曲线 $B_d(t_d)$。给定放置
$\mathcal P$，$t_d=|\mathcal P\cap\mathcal C_d|$，则：

$$
C_{LLC}(\mathcal P)=\sum_{d:t_d>0}C_d,
$$

$$
B_{LLC}(\mathcal P)=
\begin{cases}
B_d(t_d), & |\{d:t_d>0\}|=1,\\
\min\left(\sum_dB_d(t_d),B_{LLC,rank}^{sat}\right),
& |\{d:t_d>0\}|>1.
\end{cases}
$$

rank LLC 曲线的中间点可能来自 CPU-list 前缀、只激活一个域，因此不能作为任意
跨域同线程数的 cap；只用其最终饱和值表示多域汇聚 fabric 上限。DRAM 仍是整个
NUMA rank 的共享资源 $B_{DRAM}(\sum_dt_d)$，绝不按 LLC 域相加。未携带 placement
的现有 DAG 继续使用 rank 聚合曲线；只有显式 `active_cpu_ids` 或 per-domain 线程
计数的分析启用上述拓扑路径。该限制不会扩大 planner 候选，但在后续 placement
进入 task schema 前，也不能声称当前 production scoring 已完整利用多 LLC 拓扑。

主计算资源为 `gemm_core_flops`：M12、packed-A/packed-B、完整 A/B load 与
BFMMLA K-loop、无 store/epilogue 的 L1-hot GEMM。它已经联合包含 matrix issue、
frontend 和 L1-to-register 供给，不能再与这些分量分别收费。register-only
`matrix_flops`、推导出的 frontend instruction/s 和 B-only L1 byte/s 继续写入
profile 作为诊断，但不作为独立 planner 资源，也不能替代 L1-hot peak。缺少
`gemm_core_flops` 的 calibration 直接拒绝加载，不再回退到 register-only 数值。
L2/LLC/DRAM 仍由 B-only endpoint-to-register probe 测量，并分别使用
`l2_bytes/llc_bytes/dram_bytes`；深层 endpoint 已包含下游路径，因此这些时间下界
取最大值而不是相加。epilogue element/s 可选。校准还包含 cache 容量/有效容量比例、固定
call/expert/stage/range 开销、每 route 非 GEMM 开销，以及最多两个接近 1 的
W13/W2 residual scale。以上参数均与 route histogram 和 planner shape 无关。
DRAM/LLC-refill 的单核锚点必须使用与 kernel MLP 一致的 cold stream；饱和值取
持续排队开始前的 service knee，而不是 STREAM 或单次峰值。

L1/L2 热探针几何不硬编码机器容量。对 Linux sysfs 检测到的 cache 容量
$C_L$、SVE N tile $\nu$、物理 W13 最小宽度 $N_p=2\nu$ 和保留比例 $f_L$，选择：

$$
K_L=8\left\lfloor\frac{f_LC_L}{16(12+N_p)}\right\rfloor,
\qquad
Q_L=2K_L(12+N_p)\le f_LC_L.
$$

`profile_analytic_services.py` 从
`/sys/devices/system/cpu/cpuX/cache/index*` 读取 L1D/L2/LLC 容量、cache-line、
LLC id 和 `shared_cpu_list`，再由上式生成探针。rank LLC 容量是 pinned CPU 集
相交的唯一 LLC 域容量之和；LLC probe geometry 使用单域容量，不能把多个 LLC
当成一个全共享 cache。AmazonC5192Cores 上 $C_{L1}=64$ KiB、
$C_{L2}=2$ MiB、$\nu=8$；默认 $f_{L1}=0.625,f_{L2}=0.5$，得到
`M12/K728/N16` 的 `40,768 B` L1 工作集和 `M12/K18720/N16` 的
`1,048,320 B` L2 诊断工作集。比例是探针保留空间的机器无关策略参数，容量本身
必须来自硬件；CLI 可显式改变比例，但不能把某台机器的 KiB 数写入模型公式。

部署用 quick calibration 不改变上述资源定义或公式，只减少采样密度与重复次数。
service 线程点取 $\{1,2,4,8,16\}$ 中不超过 rank 核心数的点、每种 LLC 域的
半宽/全宽和 rank 全宽；planner 合法宽度仍单独声明，不要求每个候选宽度都成为
service 采样点。对同构 LLC 域只测一个代表域并按容量/核心数签名复用，异构域各补
一个 LLC-only probe；DRAM 仍只测 rank 聚合。quick profile 不做 isolated operator
residual fit，并将相对不确定性设为 20%，因此只用于 bootstrap，不替代 full
calibration 的 held-out 精度验证。校准必须由部署代码显式调用，不在 import、模型
构造或首请求中隐式执行。
部署方可用 `enable_moe_planner_quick(...)` 一次完成同步轻校准、shape-bound runtime
构造和进程注册；只有前两步全部成功才替换当前 runtime，返回后 normal MoE API 对
兼容输入直接使用 Plan V2。

上线 runtime 的第一版不在请求路径运行完整 phase-DAG 搜索。设 homogeneous team
宽度为 $t$、lane 数为 $L=C/t$，解析模型给出 isolated expert 时间
$p_i(t)=T_{iso}(M_i,t)$。对每个能整除 rank 核心数的合法 $t$，按 $M_i$ 降序将
expert 分配给使当前 lane 累计时间最小的 lane，并用

$$
\hat T_{LPT}(t)=\max_{1\le l\le L}\sum_{i\in l}p_i(t)
$$

选择最小的 homogeneous shape。该路径只生成 strict Plan V2，不搜索 mixed-width、
temporal reversal、tail pool 或 bounded tail repartition；目标是将首次未命中规划从
秒级完整 Python phase simulation 限制到少量 $T_{iso}$ 与 LPT 操作。它是可解释的
部署 bootstrap，不是完整模型的性能 oracle；完整 planner 仍用于离线比较和后续
native analytical scorer 的验证。

该 homogeneous LPT 的 production 实现保持上述决策语义，但不再通过通用
mixed-width assignment 逐 lane 重算 $p_i(t)$。对每个候选宽度先按 distinct route
count 建立 $p(M,t)$ 表，再维护 `(lane_load, lane_id)` 最小堆。由于 Python 浮点加法
可能使不同 load 在加上同一大 $p_i(t)$ 后舍入成相同 score，实现会展开堆顶所有
满足 `lane_load + p_i(t) == min_score` 的 lane，并按最小 lane id 选择；这与原有
`min(range(L), key=load+p_i)` 的 tie-break 逐位一致，不能简化成只比较未加 cost 的
load。候选集合、剪枝、目标函数、不确定性、early-merge gate 和 Plan V2 lowering
均不改变。每个 shape 的解析预测次数由约 $E(L+2)$ 降为 distinct route 数 $U$，
常规 assignment 从逐 expert 扫描全部 lane 改为 heap 操作；舍入同分桶的最坏情况
仍可展开全部 lane，但不会重复执行解析 cost model。

第二阶段把 homogeneous heap LPT、候选评分和最优 shape 选择移到单线程 C++，但
解析模型仍在 Python 中计算每个 distinct `(routes, width)` 的精确 $T_{iso}$，并把
不可变 cost rows 传给 native planner。C++ 只完整转换胜出候选的 task/dependency
payload；其余候选只返回 ranking 所需摘要，避免 pybind 为未选中的候选构造 Python
task tuples。cache miss 和已选 shape 的 cache-hit materialization 都使用同一 native
assignment；扩展不可用时保留逐位等价的 Python heap fallback。该边界迁移不改变
$T_{iso}$、候选集合、LPT 顺序、浮点同分规则、目标函数、Plan V2 schema 或默认
dispatch，且本阶段固定为一个 planner worker；候选并行属于独立的后续阶段。

quick planner 也支持用现有 `FUSED_CPP_MOE_PLANNER_THREADS` 或构造参数显式并行
homogeneous 候选。实现复用固定 candidate-index 的 `ParallelFor`：每个 worker
独立运行一个候选的确定性 heap LPT，只写预分配 slot，join 后按原始候选顺序选择
和生成 ranking。未设置线程配置时 quick planner 保持 1 worker；显式 `0` 才采用
hardware-bounded auto workers。m5 实测 2/4/8 workers 均未达到 10% 边界优化门槛，
因此多线程只保留为诊断能力，不成为 production quick 默认。

analytical DAG 的资源公式仍按上文 8 个固定 resource 顺序定义，但 production
实现不再为每个 active phase、每个 resource 重复构造 Python demand/time dict。
每个 phase/event 一次生成同序 demand/time tuple，contention loop 用固定下标读取，
并缓存 immutable phase 的 isolated `base_ns`。scalar `resource_demand()`、
`resource_times_ns()`、pressure 字段和 explain event 继续保留；vector tuple 只是
相同公式的数据布局。归约仍使用相同 Python `sum/max` 顺序，spill fraction、service
capacity、dilation、event 边界、completion epsilon 和 early-merge 三态均不改变。

设某 stage 有 $P$ 个物理 M panel；第 $j$ 个顺序 N range 的 packed-B 字节为
$B_j$、每个 owner 的 B 窗口为 $U_j$、active owner 数为 $t_j$；全部物理
packed-A 字节为 $A$，最大单 panel packed-A 字节为 $A_p$。range 按 tile 数
非递增分配，因此后续 active owners 是首个 range 的子集。packed-B 使用独立
repeated-scan probe 的有效容量比例
$\rho_2^B$ 与三点保留率校准 $g_2(W)$：低于拐点的 miss floor、名义 L2 容量处
miss、两倍容量处 miss，并在两段之间 smoothstep 插值：

$$
C^A_{2,\mathrm{eff}}=\rho_2^A C_2,
\qquad
C^B_{2,\mathrm{eff}}=\rho_2^B C_2.
$$

$\rho_2^A$ 与 $\rho_2^B$ 不要求相等；packed-B stripe 与 A、store 和 prefetch
状态竞争，实测 retention knee 可以显著早于通用 L2 容量边界。$g_2$ 只描述 A
尚未完成一个 effective-L2 turnover 时的**瞬态** repeated scan。定义：

$$
L_A=\max\left(\left\lfloor\frac{C^A_{2,\mathrm{eff}}}{A_p}\right\rfloor,1\right),
\qquad
q=\min(P-1,L_A),
$$

$$
r_B(W)=\mathbf 1[W>C_2],
\qquad
r_A(U_j,A)=\mathbf 1[U_j+A>C_2-A_p].
$$

$L_A$ 是在 owner 前流过一个 effective L2 所需的 A panel 数；之后 B 进入物理
驻留/流式稳态。$r_A$ 为 kernel 的在途 load/prefetch 保留一个 $A_p$ headroom，
避免把正好贴住 L2 的 A 错判为可跨 range 驻留。于是：

$$
Q_{B,L2}
=\sum_j B_j\left[
1+qg_2(U_j+A_p)+(P-1-q)r_B(U_j+A_p)
\right],
$$

$$
Q_{A,L2}
=A\left[t_1+\sum_{j>1}t_jr_A(U_j,A)\right].
$$

这直接表达当前 kernel loop：首次 B scan 是冷权重；前 $q$ 次复用由实测瞬态
retention 决定，之后能装进物理 L2 的 B stripe 驻留、超容量 stripe 继续流式；
每个 N owner 至少扫描一次 A，后续顺序 range 只有在完整 A、owner B 与在途 panel
共同可驻留时才不产生新 refill。旧公式把一次短 route 的残余 miss 乘到所有
$P-1$ 个 panel，在长 route 上会无界放大并系统性偏向小窗口；新 turnover 长度完全
由 cache/panel 几何得到，不引入 route 阈值或时间表。对 production 的 distinct-expert 流式访问，
compulsory DRAM 只有首次 packed-B：

$$
Q_{\mathrm{DRAM}}^{\mathrm{comp}}=\sum_j B_j.
$$

packed-A 和 W13 intermediate 由同一 operator 刚刚生成，默认从 cache hierarchy
供给；A refill、重复 B refill 和 C writeback 都记为 spillable。若同时活跃 phase
的工作集为 $\sum_i W_i$，使用 effective/physical LLC 间的 $h_3(\sum_iW_i)$
决定这些 spillable bytes 中进入 DRAM 的比例。

$W_i$ 中只有 $P>1$ 时才包含 active B window。$P=1$ 的 M<=12 B 仍计入
compulsory DRAM 和 stream bandwidth，但没有后续 panel reuse，不占
`reusable_B_capacity`；A 和 C 因其他 owner/下一 stage 仍会消费而继续进入
工作集。这与 9.6 的 M12 冷 B 扩展和 one-pass LLC pollution 结论一致。

每个顺序 N range $j$ 进一步按物理执行拆为：零资源流量的
`range_setup`、只执行第一个 M panel 的 `cold_b`，以及在 $P>1$ 时执行其余
$P-1$ 个 panel 的 `steady_b`。设首 panel 的 compute/store rows 占比分别为
$f_{c,j},f_{s,j}$，则：

$$
Q^{cold}_{B,\mathrm{DRAM}}=B_j,\qquad
Q^{steady}_{B,\mathrm{DRAM}}=0,
$$

$$
Q^{cold}_{A,L2}=f_{c,j}Q_{A,L2,j},\qquad
Q^{cold}_{C}=f_{s,j}Q_{C,j},
$$

其余 A/C 与 $Q_{B,L2,j}-B_j$ 全部进入 `steady_b`。matrix FLOPs、frontend
instructions、L1 load 和 epilogue elements 则直接按首 panel 与剩余 panel 的
真实 Mr/K/N tile 指令数生成，而不是按 wall time 比例拆分。每个 range 的两类
phase demand 求和严格等于原 kernel demand；$M\le12$ 没有 steady phase。

每个有效校准都必须存在 `gemm_core_flops`，cold/steady phase 使用：

$$
T_{\mathrm{core}}
=\frac{F^{\mathrm{bal}}}{P_{\mathrm{gemm\_core}}(t)},
$$

$$
T_{\mathrm{xfer}}
=\max\left(
\frac{Q_{L2}}{B_{L2}(t)},
\frac{Q_{LLC}}{B_{LLC}(t)},
\frac{Q_{\mathrm{DRAM}}}{B_{\mathrm{DRAM}}(t)}
\right),
$$

$$
T_{\mathrm{body}}
=\max\left(
T_{\mathrm{core}},
T_{\mathrm{xfer}}
\right),
$$

stage/range fixed cost 单独作为 setup phase，epilogue service 仍串行加在对应
cold/steady phase 后。isolated stage 为：

$$
T_s=\sum_j\left(T_{setup,j}+T_{cold,j}+T_{steady,j}\right),
$$

其中单 panel range 的 $T_{steady,j}=0$。缺少 `gemm_core_flops` 的旧 calibration
不再兼容，因为 register-only BFMMLA 不是 kernel 可达到的 peak。先分 phase 再取
ECM maximum 很重要：cold panel 可以由
DRAM/refill 主导，而同一 range 的 steady 部分可以由 GEMM core 或 private-cache
主导。第 $j$ 个 range 的 balanced
N-tile demand 为
$\lceil n_j/t\rceil\min(n_j,t)$；因此 N tile 不能整除、尾 range 少于 team
width、非 2 次幂宽度和 active-thread 截断都不需要额外 route table。

并发事件中先按当前 active working set 的 $h_3(\sum_iW_i)$ 计算 spill。令
$\tau_{i,r}$ 为 phase $i$ 对资源 $r$ 的独立 service occupancy（包含 stage
residual），只统计对该资源有非零请求的线程：

$$
\lambda_r=\sum_i\frac{q_{i,r}}{\tau_{i,r}},
\qquad
c_r=\min\left(C,\sum_i t_i\mathbf 1[q_{i,r}>0]\right),
\qquad
\rho_r=\frac{\lambda_r}{R_r(c_r)},
\qquad
d_r=\max(1,\rho_r).
$$

$d_r$ 只放大对应 GEMM core、L2、LLC、DRAM 或 epilogue 分量；分配后
速率 $\lambda_r/d_r$ 不超过容量。固定开销不作为内存流量 derate，
event simulator 在 setup/cold/steady completion 处重算 active set 和所有 $d_r$。
因此不需要 task-pair slowdown matrix：短 expert
几乎全部位于 cold phase，共享 refill 服务降低会直接减慢整体；长 expert 大部分
位于 steady compute/cache phase，同一争用只影响部分 lifetime，于是自然得到有
方向性的交叉减速。`explain_dag()` 输出每个事件的工作集、spill fraction、
offered/allocated rate、capacity、utilization 和 dilation，且与 planner 评分使用
同一推进器。

解析 backend 与 empirical backend 共用 planner protocol。前者根据机器允许
宽度生成 homogeneous/至多两种宽度的 shape，后者继续严格使用 profile shape。
在 9.14 的真实机器验收门槛通过前，production 默认仍为 empirical backend；
解析 backend 先用于显式 shadow/what-if 规划。完整 schema、假设和运行命令见
`cost_model/ANALYTIC_MODEL.md`。

#### 8.2.5 Full-stage packed-B 复用与 width-derived owner stripe

当前 production 对每个 stage 只有一个 full-N tile domain。沿用 1.1 的记号，
单 tile 字节数、活跃 owner 数和最忙 owner stripe 为：

$$
b_s=2K_s\nu,\qquad
a_s(t)=\min(t,q_s),\qquad
u_s(t)=b_s\left\lceil\frac{q_s}{t}\right\rceil.
$$

首个 M panel 对 packed B 的 compulsory 流量为完整 $B_s=b_sq_s$；后续 panel
是否从 private L2/LLC 重用由 $u_s(t)+A_{p,s}$ 与 cache capacity 决定。N-split
下每个活跃 owner 扫描完整 packed A，因此 stage 的 owner 聚合 A 边界流量基线为

$$
Q_{A,s}=a_s(t)A_s,qquad A_s=2M K_s.
$$

这两个量都由 $(M,t,K_s,N_s,\nu)$ 唯一确定。planner 只能通过选择 $t$ 同时改变
GEMM 并行度、A 扫描数和 owner stripe；不能在固定 $t$ 下再拆 weight stage。
cost model 的 active shared working set 使用完整 $B_s$，private-cache 诊断使用
$u_s(t)$，不得把二者混为一个 byte-window。

### 8.3 Full-stage owner-cache 工作集 band

对并发 active task 集 $\mathcal A(\tau)$，stage working set 与最大 owner stripe
分别记录为：

$$
S_{active}(\tau)=\sum_{i\in\mathcal A(\tau)}B_{s_i},
\qquad
U_{max}(\tau)=\max_{i\in\mathcal A(\tau)}u_{s_i}(t_i).
$$

owner-cache shadow 仍可用独立 weight-scan 推导并发 stream band，但一个 expert 的
stream size 必须是完整 stage，不是人为 range。若每核 private L2 为 $L_2$、组相联
路数为 $A$、预留 $r$ 个 way，可用总 owner budget 为

$$
C_{owner}=CL_2\frac{A-r}{A},
$$

并以严格不等式 $nB_s<C_{owner}$ 得到容量上界。该 band 只用于候选排序/验证，
不是执行硬约束；最终仍由 contention-aware event simulator 评分。profile 必须
声明 `stage_geometry=full_n_team_stripes`，旧非单位 range calibration 直接拒绝，
不能通过重命名复用。

#### 历史：packed-B range/window 模型（非规范）

以下直到 8.4 的内容记录 2026-08-09 之前的 range/window 实验、反例和演进，
只用于解释历史结果。其 $R_s$、$g_s$、$g_\theta$、stage-window policy 与 Plan
字段均已从 production 实现删除；若与 1.1、8.2.5 或本节冲突，以 full-stage
公式为准。

8.2.2 的保守映射取 $Q_{\mathrm{shared},B}=2KNP$，即假设每个 M panel 都重新扫一
遍 packed-B；9.23 的四状态计数取相反极限，假设 B 在后续 panel 上恒为 hot
（$P-1$ 次 `ch` 加 $(P-1)(Q-1)$ 次 `hh`）。两者都是端点，真实取值由**每线程
瞬时 packed-B 窗口**决定，而该窗口同时受 team 宽度和 stage window 目标控制。

设某个 stage window 的 packed-B 为 $W_s$ 字节，planner 选定的 window 目标为
$g_s$，执行该 window 的 team 宽度为 $t$。kernel 按 N 轴把单个 range 切给队内
线程，因此定义

$$
\omega(g_s,t)=\frac{g_s}{t},
\qquad
R(g_s)=\left\lceil\frac{W_s}{g_s}\right\rceil,
$$

即**每线程窗口**与 **range 数**。有效重读因子定义为实测有用带宽相对该机器
峰值有用带宽的倒数：

$$
p_{\mathrm{eff}}
=\frac{B^{\ast}}{B_{\mathrm{useful}}(M,\omega,t)},
\qquad
1\le p_{\mathrm{eff}}\le\left\lceil\frac{M}{12}\right\rceil .
$$

上界在 $\omega$ 显著超过私有 L2 且并发活跃窗口超过 LLC 时取到；$M\le12$ 时
$P=1$，上界退化为 $1$，此时窗口不影响流量。9.24 在 $M=28$ 上实测
$p_{\mathrm{eff}}$ 随 $\omega$ 从 $4$ MiB 降到 $0.25$ MiB 时由 $2.97$ 单调降到
$1.22$，并在 $\omega\le0.25$ MiB 后进入平台；$M=12$ 的对照在同一 $\omega$ 域内
变化不超过 $0.7\%$。

上述 $p_{\mathrm{eff}}$ 由墙钟带宽反推，可用 PMU 直接核对。`l2d_cache_refill`
统计被填入 L2 的行数（含硬件预取），乘 64 即跨越 L2 边界的字节数。$M=12$ 的
对照给出该口径的标定：跨 L2 字节数为必需 packed-B 的 $1.02$ 倍，即每字节恰好
跨一次。$M=28$、$192$ expert、`24x4T` 下扣除不随 $R$ 增长的 A 与 C 之后：

| $\omega$ | 跨 L2 / 必需 B | 反解 $p_{\mathrm{eff}}$ | 墙钟 $p_{\mathrm{eff}}$ |
| ---: | ---: | ---: | ---: |
| 1 MiB | 1.87 | 1.83 | 1.75 |
| 0.25 MiB | 1.19 | 1.15 | 1.22 |
| 0.0625 MiB | 1.19 | 1.16 | 1.23 |

两者吻合在 $5\%$ 内，故 $p_{\mathrm{eff}}$ 就是 packed-B 在 L2 边界上的重复搬运
次数，机制为私有 L2 驻留而非 DRAM 带宽。（`ll_cache_miss_rd` 只数需求缺失、
不含预取，$M=12$ 仅 13 MB 而实际搬运 2.4 GB，故不能用作 DRAM 流量；
`l3d_cache_refill` 在 Neoverse-V3 上返回 0。因此 LLC 与 DRAM 之间那一段仍未量
到，但与分离实验对 LLC 容量假设的否证方向一致。）

大 $M$ 处于相反区间，且主导项不是 B。$M=2040$、24 expert、`24x4T` 下把
$\omega$ 从 $1$ MiB 压到 $0.0625$ MiB，跨 L2 流量由 $6.13$ GB 涨到 $50.37$ GB
（$8.2$ 倍）；$\omega=1$ MiB 处该流量已是必需 packed-B 的 $21.8$ 倍，说明主体是
随 $M$ 线性增长的操作数被每个 range 重扫：

$$
\text{每 range 的 A（全队）}=2MK_s,
\qquad
\text{每 range 的 B}=g_s\ \text{（与}\ M\ \text{无关）}.
$$

这里的 $2MK_s$ 是**一个 range 内全队合计**的 A 流量，两种 team 分割几何下都成立，
但每线程的份额不同（见 9.29）：`kN` 下每线程读全部 $M$ 行、全队为 $t\cdot 2MK_s$；
`kM` 下每线程读 $M/t$ 行、全队为 $2MK_s$。本节的 $M=2040$、`24x4T` 数据 W13 走
`kM`，故 $2MK_s$ 即全队量。

$M=28$ 的 W13 shared-A 为 $229$ KiB，装得进私有 L2，跨 L2 只搬一次、不随 $R$
增长；$M=2040$ 的全队量为 $16.7$ MB（`kM` 下每线程 $3.98$ MiB），装不进 L2，于是
代价正比于 $R$。由此 $\omega^\ast$ 随 $M$ 单调上升：短 route 侧收益大代价小，长
route 侧收益已接近零而代价随 $R$ 线性增长。定量核对：$M=2040$ 从 $\omega=1$ 到
$0.25$ MiB 时 A 重扫增量预测 $+6.9$ ms，实测墙钟 $+6.18$ ms，误差 $12\%$。该核对
同时**判别了几何**：按 `kM` 算增量为 $2.41$ GB（$\approx6.9$ ms），按 `kN` 算为
$9.63$ GB（$\approx27.5$ ms），与实测差 $4.4$ 倍，故实测独立确认 W13 在此点走 `kM`。

关键结论是 $p_{\mathrm{eff}}$ 由 $\omega$ 而不是由 $t$ 或 $g_s$ 单独决定。因此
**加宽 team 与缩小窗口是达到同一 $\omega$ 的两条可互换路径**，在相同 $\omega$ 下
四种宽度的实测差异只有 $3.0\%$--$5.6\%$。两者仍不完全等价，因为

$$
R(g_s)\ \text{只随}\ g_s\ \text{变化，不随}\ t\ \text{变化},
$$

而每个 range 都要重跑完整 M-panel 循环，产生一份与 $R$ 成正比的固定成本
（range dispatch、A 重扫、team barrier、N-tile 收尾）。$t$ 则通过并发 expert 数
$C/t$ 影响可达的 memory-level parallelism。于是最优点是内部解：9.24 中
$4T$ 优于 $1T$ 共 $5.5\%$、优于 $8T$ 共 $3.1\%$。

$\omega$ 描述的是**私有**驻留而非共享 LLC 容量。全核忙时聚合活跃窗口恒为
$C\omega$，因此只扫 $g_s$ 无法区分两级；固定 $t$ 改变活跃核数 $C$ 可以分离。
9.24 的分离实验显示：同一聚合窗口下 $p_{\mathrm{eff}}$ 相差 2 倍且严格跟随
$\omega$，而固定 $\omega$ 时聚合窗口变化 8 倍只使 $p_{\mathrm{eff}}$ 变化
$1.36$--$1.77$ 倍**且方向与 LLC 容量假设相反**。因此共享 LLC 容量被否证，
$g(M,t)$ 无需引入活跃核数项；实测有效驻留容量约 $0.25$ MiB/线程，即标称
$2$ MiB 私有 L2 的约 $1/8$，与 packed A、intermediate 和 C 输出穿透同一 L2 一致。
残余的 $g(C)$ 项（核数越多 $p_{\mathrm{eff}}$ 越好）机制未定，但符号方向使
96 核标定在更少活跃核时偏保守。

该修正只改变 8.2.2 中 $Q_{\mathrm{shared},B}$ 与 9.23 四状态计数的**实现条件**，
不改变算法层公式、planner 候选空间、宽度剪枝或 $g(M,t)$ 的确定性性质。当前
production 仍使用实测 shape derate 隐式吸收该效应；把 $p_{\mathrm{eff}}(\omega)$
显式写进 demand 之前，需要按 W13/W2 分离标定并通过 9.14 的 contention/regret
门槛。

**参数化推论。** 既然不变量是 $\omega$ 而不是 $g_s$，stage-window policy 的输入
单位就应当是 $\omega$，由 policy 在已知 $t$ 处一次性下降为 $g_s$：

$$
g_s(\omega,t)=b_s\left\lceil\frac{q_s}{R}\right\rceil,
\qquad
R=\min\left\{R:\ b_s\left\lceil\frac{\lceil q_s/R\rceil}{t}\right\rceil\le\omega\right\},
$$

其中 $b_s,q_s$ 沿用 8.3 的单 tile 字节数与总 tile 数。该下降与 8.3 的
$\widehat S_s$ 精确互逆，因此 plan、kernel ABI 与 native planner 仍只见整数
$R$ 与 $g_s$，不引入浮点。可达 $\omega$ 因此量化为 $b_s$ 的整数倍，下界
$b_s$、上界 $W_s/t$；$T$ 越大可达集越密，这是宽度与窗口存在联合量化耦合的根源。

这个换单位使已标定的 `amazon_c5_192c_tp4_f512_v1` 表从「4 route band $\times$ 4
宽度 $\times$ 2 stage $=26$ 个 $g_s$」塌缩为「4 band $\times$ 2 stage $=8$ 个
$\omega$」加 6 个偏差格：band `144--287` 的四种宽度精确同为
$\omega_{W13}=1/8$ MiB，band `288--575` 有 3/4 宽度同为
$\omega_{W13}=1/2,\omega_{W2}=1/8$ MiB。这张表当初是按 $(\text{band},t)$ 逐格独立
搜索得到的，它自己收敛到常数 $\omega$，是 $\omega$ 为不变量的独立证据。6 个偏差
格全部只差一档 factor-2 且集中在 $t=1$ 与 $t=8$ 的 W2，与本节测得的两端不变性
最弱一致。

两个 stage 的 $\omega^\ast$ 不必相等。$W13$ 的 shared-A 为 $2MH$，$W2$ 为 $2MF$，
相差 $H/F=8$ 倍，因此在 A 重扫成为主导项的长 route 侧，W2 能承受更多 range、
最优 $\omega$ 更小。9.24 的二维扫描证实了这个方向但界定了适用域：

| $M$ | 最优 $(\omega_{W13},\omega_{W2})$ MiB | 该 $\omega_{W13}$ 下 $\omega_{W2}$ 的极差 | production band |
| ---: | :--- | ---: | :--- |
| 13 | $(1/8,\ 1/4)$ | 0.67% | $(1/4,1/4)$，差 $-1.36\%$ |
| 28 | $(1/4,\ 1/4)$ | 4.81% | $(1/4,1/4)$，**精确命中** |
| 48 | $(1/8,\ 1/4)$ | 0.69% | $(1/4,1/4)$，差 $-2.95\%$ |
| 120 | $(1/8,\ 1/8)$ | 9.53% | $(1/8,1/8)$，**精确命中** |
| 320 | $(1/2,\ 1/8)$ | 1.52% | $(1/2,1/8)$，**精确命中** |

两点结论。第一，$\omega_{W13}$ 是强轴、$\omega_{W2}$ 是弱轴：固定最优
$\omega_{W13}$ 后 $\omega_{W2}$ 在 3--4 个档位上的极差多在 $1.5\%$ 以内，$M=28$ 与
$M=120$ 的较大极差全部来自 $\omega_{W2}=1/2$ MiB 处的悬崖，而 $\omega_{W13}$ 在
每个 $M$ 上都有清晰的内部峰且偏离一档要损失 $3\%$--$30\%$。第二，$H/F$ 论证只在
$M=320$ 成立（$4$ 倍差），$M=120$ 两者相等，而 $M\le48$ 时 W2 反而偏好**更大**的
$\omega$——此时两个 stage 的 shared-A 都只有 $13$--$393$ KiB、都装得进私有 L2，
A 重扫项对两者均可忽略，短 route 侧决定 $\omega_{W2}^\ast$ 的机制未识别，但效应
不超过 $1.5\%$。

旧 split/no-split 由此只对应两个 range 端点：W13 总量 $8$ MiB 时
$g_{W13}=4$ MiB 即 $R=2$，$g_{W13}=8$ MiB 即 $R=1$，且
$W2$ 总量 $4$ MiB 使任何 $g\ge4$ MiB 都退化为单 range。profile identity 记录的
是 achieved $R$，本身无单位；runtime 和 catalog 不再保留旧编码。

#### 历史续：Stage-range owner-cache 工作集 band

当前 production 基线几何为 W13 两个相等 N range、W2 一个 range，即
$(R_{13},R_2)=(2,1)$。packed 布局与 range 数无关；range 只改变遍历顺序。
cache 模型可以先用字节目标 $g_s$ 表达容量需求，但必须在 planner 中量化为精确
正整数 $R_s$，profile 和 runtime 均不接收字节目标。
对 BF16 的 $H,F$，三个顺序 packed-weight stage 都是：

$$
S_0=2HF\quad\text{bytes}.
$$

解析模型可构造 packed-B byte-window 候选。设 stage $s$ 的 packed
维度为 $(K_s,N_s)$，BF16 N tile 为 $\nu$，总 tile 数和单 tile 字节数为：

$$
q_s=\frac{N_s}{\nu},\qquad b_s=2K_s\nu.
$$

给定目标 $S_{\mathrm{target}}>0$，实现选择：

$$
u_s=\max\left(1,\left\lfloor\frac{S_{\mathrm{target}}}{b_s}\right\rfloor\right),
\qquad
r_s=\left\lceil\frac{q_s}{u_s}\right\rceil,
$$

再将完整 N tiles 均匀分配给 $r_s$ 个顺序 range。最大实际 range 为：

$$
\widehat S_s
=2K_s\nu\left\lceil\frac{q_s}{r_s}\right\rceil.
$$

当 $S_{\mathrm{target}}\ge b_s$ 时有
$\widehat S_s\le S_{\mathrm{target}}$；否则最小粒度是一个 N tile，实际 range
会大于目标。W13 使用 $(K,N)=(H,2F)$，W2 使用 $(F,H)$。例如 TP4
$H=4096,F=512,\nu=8$ 时，legacy 4 MiB stage 的 W13/W2 range 数为 $2/1$，
2 MiB 为 $4/2$，1 MiB 为 $8/4$。

每个 range 内仍对 N tiles 做现有线程切分，因此单 range 的有效线程上界为：

$$
t_{\mathrm{active},s}
\le
\min\left(t,\left\lceil\frac{q_s}{r_s}\right\rceil\right).
$$

更小窗口不改变 GEMM useful/executed FLOPs、aggregate BFMMLA 或 aggregate packed-B
读取，只增加 range dispatch，并在 8.2 的保守模型中增加 shared-A range scan。
相邻 range 之间没有 barrier；不同 worker 可以短暂处于相邻 range，因此
$\widehat S_s$ 是 loop-order 的 nominal active window，不是同步保证的硬 cache
residency 上界。W2 owner-scatter 必须按每个 range 复用相同的 discontiguous N
ownership，才能继续消除 W2-to-scatter barrier。

N-split 中每个 worker 持有互不重叠的 packed-B column slice，因此决定首次
容量拐点的是 owner-private cache，而不是只看 shared LLC。设可用核心数为
$C$，每核 private L2 为 $L_2$、组相联路数为 $A$，为 packed-A、store、
prefetch 与 replacement headroom 预留 $r$ 个 way，则保守 owner budget 为：

$$
C_{\mathrm{owner}}=CL_2\frac{A-r}{A}.
$$

使用严格容量不等式 $nS<C_{\mathrm{owner}}$，可得上界：

$$
n_{\max}=\left\lceil\frac{C_{\mathrm{owner}}}{S}\right\rceil-1.
$$

无 GEMM 的 owner-slice scan 使用饱和曲线：

$$
B_{\mathrm{owner}}(n)
=B_{\max}\left(1-e^{-n/n_s}\right).
$$

若要求达到峰值比例 $u$，并发 stream 下界为：

$$
n_{\min}=\left\lceil-n_s\ln(1-u)\right\rceil.
$$

因此可解释的工作集候选 band 是
$n\in[n_{\min},n_{\max}]$。对 band 内 shape $\sigma$，令完整 measurement
window 中 lane $l$ 有 $k_l$ 个 expert，使用：

$$
T_{\mathrm{iso-call}}(M,\sigma)
=\max_l k_lT_{\mathrm{iso}}(M,t_l).
$$

先保留不超过 band 内最佳 isolated makespan $(1+\epsilon)$ 的 shape，再选
工作集最小者，以便在算力近似相同时保留 cache headroom。该规则目前只对
physical M12 panel 数不少于 16 的长 route 生效，并且仍是 shadow validation，
尚未改变 production planner 的 shape 可行域。完整实验见
`cost_model/WORKING_SET_MODEL_VALIDATION.md`。

owner-cache band 是对 $\widehat D_i(\mathcal Z)$ 和候选空间的实现相关近似，
不是 LLC 硬可行性约束。推荐的求解分层是：

1. 使用只有 CPU 容量、$D_i\equiv1$ 的 CP-SAT 生成 isolated lower bound 和候选；
2. 使用 owner-cache band 等规则缩小需要评估的 active configurations；
3. 使用 contention-aware event simulator 对候选重新评分；
4. 在线 planner 使用经真实 runtime regret 验证过的低开销启发式。

令 empirical 校准域为

$$
d=(\text{machine/topology},\text{distributed expert shape},
   \text{kernel implementation/tail},\text{source/binary hash}).
$$

每个域只允许一个活动 profile，并校准

$$
\widehat I_i^{(d)}(t),\qquad
\widehat D_i^{(d)}(\mathcal Z).
$$

profile 仍必须显式记录采样时使用的
$g_{cal}=(R_{13}^{cal},R_2^{cal})$，用于解释表中时间、拆分 phase 和判断
full-call anchor 是否适用；它是 measurement provenance，不属于 $d$，也不进入
`ProfileQuery`、plan cache 或 planner 候选。旧 `w13_split`、
`w13_split_chunks` 和 byte target 即使留在历史 JSON 中也不会被读取。catalog
若在同一 $d$ 下看到第二份 profile 会直接报 duplicate calibration，而不会把两份
range 几何变成候选。

在线 planner 只搜索已有 schedule shape：

$$
\sigma^*=\arg\min_{\sigma\in\Sigma_d}
\widehat C_d\left(\sigma;
\{g_\theta(M_i,t_i(\sigma))\}_{i\in\mathcal J}\right).
$$

Plan V2 为每个 whole-expert task 显式携带两个 stage range 数：

$$
(R_{13,i},R_{2,i})=
g_\theta(M_i,t_i),\qquad R_{13,i}\ge1,\quad R_{2,i}\ge1.
$$

其中 $M_i$ 是 route 数，$t_i$ 是 task 实际执行宽度；每个正整数都是 runtime
必须执行的精确、tile-aligned range 数。Plan V2 不再存在 `-1` 继承；legacy plan
bridge 在升级时显式写入 `1/1`。tail-pool task 使用 pool 的实际宽度，而不是原
strict head 宽度。$g_\theta$ 是按机器、NUMA、shape 和 kernel identity 命名的
确定性策略。对每个已有候选 $\sigma$，
planner 先由候选确定各 task 的 $t_i$，再唯一解析 $g_\theta(M_i,t_i)$，之后才
计算：

$$
\widehat C_{d,\theta}(\sigma)
=\widehat C_d\left(\sigma;
\{g_\theta(M_i,t_i)\}_{i\in\mathcal J}\right).
$$

因此候选集合只有 $\Sigma_d$ 加已有 tail-pool/bounded-tail 候选，没有
operator-wide $(1,1)/(2,1)$ 或其他 range 枚举维度。由 $R_{s,i}$ 和 stage 总
tile 数可唯一恢复 $\widehat S_{s,i}$。nominal active packed-B 诊断改为

$$
S_{active}(\sigma)=\sum_{i\in active}
\max(\widehat S_{13,i},\widehat S_{2,i}),
$$

每个 task 的 owner-private stripe 为

$$
S_{owner,s,i}=b_s\left\lceil
\frac{\left\lceil q_s/R_{s,i}\right\rceil}{t_i}
\right\rceil.
$$

stage simulator 按实际 task range 推进瞬时 $\sum_i\widehat S_{s,i}$。plan cache
包含校准域/digest、策略名称和完整 route signature；逐 task range 在候选宽度确定后
重新生成并随 Plan V2 保存，不再缓存或输出一个全局 pair。

公开 Python/C++ ABI 只接受 exact range：同步、scheduled 和 legacy async 入口
分别接收正整数 `w13_ranges`/`w2_ranges`，Plan V2 接收逐 task 的正整数张量。
布尔 split、全局/per-task byte-window、环境变量回退和 Plan V2 legacy native
fallback 均已删除。W2 GEMM 与 owner-scatter 必须使用同一个 $R_{2,i}$；非法或
超过 stage tile 数的 range 由 Python/native 边界拒绝。

对 empirical phase model，现有 isolated table 仍校准于 $g_{cal}$；在没有
独立 window-isolated residual 前，$\widehat I(M,t)$ 保持原表值，不凭空外推
range dispatch 的单独修正。但该总时间会按实际
$(r_{13,i},\widehat S_{13,i},r_{2,i},\widehat S_{2,i})$ 拆成 W13/W2 range
phase，contention simulator 使用每个时刻真实 active phase 的
$\sum_i\widehat S_{s,i}$ 计算 equivalent working set。因而策略带来的并发
cache-pressure 变化进入 makespan；单 task 的绝对时间仍受原 isolated
calibration 限制。解析 backend 则直接用实际 geometry 重算 range overhead、
owner L2/LLC miss、DRAM demand 和 $\widehat I(M,t)$。

对解析 backend，$g_\theta$ 不再来自 route band 表，而由同一物理 demand 模型直接
生成。stage $s$、route $M$、已选 team width $t$ 的候选集合
$\mathcal G_s(t)$ 只包含运行时可实现的 tile-aligned range，并施加：

$$
\omega_s(g,t)\ge C_{L1},
\qquad
n_{s,\mathrm{range}}(g)\ge\min(t,q_s),
\qquad
n_{s,\mathrm{owner}}(g,t)\in\{1,2,4,\ldots\},
$$

自然解析点应用上述三个约束。除此之外，候选集显式保留可比较的几何端点：

$$
\mathcal G_{W13}(t)=\{g(R=1),g(R=2),g_{cal}\}\cup\mathcal G^{natural}_{W13}(t),
$$

$$
\mathcal G_{W2}(t)=\{g(R=1),g_{cal}\}\cup\mathcal G^{natural}_{W2}(t).
$$

第一项约束禁止把 owner stripe 压到 L1D 以下后继续增加控制开销；第二项禁止
range 内 tile 少于可用 team 线程而造成 barrier 自旋；第三项限制为 kernel 自然
二分层级。端点用于覆盖旧两种编码和做 shadow 对照，不再把旧 split 值当作
$R_{W13}$ 的可行域下界。$M\le12$ 只有一个 physical panel，每个 B tile 只消费一次，
不存在可由更多 range 保护的 B 重读；因此 $R=1$ 在物理需求相同下严格少一份 range
控制，直接由支配关系选出，而不是继承 operator-wide split 状态。

绝对时间仍按 8.2.4 的 ECM 重叠下界
$T_{body}=\max(T_{core},T_{xfer})$。该式在 transfer 全落于同一 compute ceiling 下时
无法区分窗口，因而 policy 选择单独使用串行增量目标：

$$
J_s(M,t,g)=
\sum_{p\in\mathcal P_s(g)}
\left[
T_{core,p}+T_{epi,p}+
\max(T_{L2,p},T_{LLC,p},T_{DRAM,p})
\right]
+R_s(g)\tau_{range},
$$

令 $g_{\min}=\arg\min_gJ_s(M,t,g)$，machine calibration 的相对不确定度为
$\epsilon_{rel}$。低于

$$
\Delta_J=\epsilon_{rel}T_{xfer}(g_{\min})
$$

的候选差异不视为物理可分辨，先形成等价集：

$$
\mathcal E_s(M,t)=
\{g\in\mathcal G_s(t):J_s(M,t,g)\le J_s(M,t,g_{\min})+\Delta_J\}.
$$

令一个 packed-B tile 为 $b_s=K_s\nu\cdot2$，kernel-native owner 窗口为

$$
U_{s,pref}=\max(C_{L1D},2b_s).
$$

最终选择：

$$
g_s^*(M,t)=\arg\min_{g\in\mathcal E_s}
\left(
\left|\log_2\frac{U_s(g)}{U_{s,pref}}\right|,
J_s(M,t,g),R_s(g)
\right).
$$

它把候选相关的 refill/争用需求暴露出来，但**不替代**绝对时间模型；选择后仍用
ECM maximum 和共享资源 event simulator 评分整个 DAG。W13/W2 分别独立选择，再
量化为 exact-range ABI。不确定度 tie 只使用一个 machine-level 标量和已有
cache/kernel 几何，不引入 route band 或计时表。策略由 calibration digest 唯一命名，
planner shape、tail-pool、kernel variant 与剪枝集合均不增加。

SVE packed N tile $\nu$ 进入 machine/kernel calibration。解析模型默认使用
`kernel.backend_n_tile`，holdout benchmark 将其与 packed weight runtime ABI
比较并在不一致时拒绝测试。两台验证机分别为 AmazonC5192Cores 的 $\nu=8$ 与
AmazonECSV1 的 $\nu=16$；若仍硬编码 8，8-core 的 4T 小窗口会只给两个线程分到
N tile，测得的退化来自错误 ABI 而非解析策略。

profile 的 homogeneous full-call anchor 只有在该 route 和 shape 的所有 lane
都保持 measurement geometry $g_{cal}$ 时继续使用；任一 lane 解析到不同 range
时必须走 stage event simulation，不能拿 baseline full-call 时间覆盖新执行语义。候选的
`active_working_set_bytes` 与 per-worker owner-window 诊断也按每个 lane 实际
task range 计算。

empirical backend 当前只有同时精确匹配双 NUMA AmazonC5192Cores 的 TP4
`H=4096,F=512,E=256`、96-core rank、SVE JIT exact-M implementation identity
和 rank CPU 集合时，已在两个 NUMA rank 上验证的
`amazon_c5_192c_tp4_f512_v4` 作为默认 runtime policy；任一字段不匹配或显式
设置 `use_default_stage_window_policy=False` 时都使用 profile measurement geometry
作为未覆盖 task 的校准 fallback。该默认是
受限的确定性 execution policy；analytic backend 则在任意具有完整 machine
calibration 的 SVE 模型上生成上述 policy。两者都不代表窗口已成为 cost-model
搜索变量。特别是
单个 M12 panel 不复用 B，额外 range 通常只有 dispatch 和 lane-width 代价，
不能仅按 cache 容量规则强制细分。

#### 8.3.1 v0.88 实现验证

2026-08-09 在 macOS Arm64 本地重编扩展后，cost/profile/planner/native planner
定向测试为 `151 passed`；Plan V2、exact-range ABI、timeline 与 fused-MoE 边界为
`51 passed, 104 skipped`（SVE-only case 在 macOS 跳过）。在
AmazonC5192Cores 重新编译 Linux AArch64 SVE 扩展后，同两组测试分别为
`151 passed` 与 `154 passed, 1 skipped`。

96-core NUMA0 planner smoke 使用 canonical TP4/F512/E256 calibration：均匀
$M=48$ 的 256 个 task 全部由 $g_\theta$ 解析为 `(8,4)`；
`dsv4-real-2048-seq70` 同一个 plan 内得到
`(8,4):120, (2,1):78, (8,8):15, (2,4):10`。两个 plan 都没有
`operator_options`。这同时验证了 calibration geometry `(2,1)` 只用于未覆盖
fallback，而不是 plan-level 控制或 profile 搜索轴。

### 8.4 x86 per-expert pattern/cache 与 team-N/wave policy

x86 AMX backend 不把 pattern/cache choice 加入 CPU MoE planner。设 expert $i$ 的
route 数为 $M_i$，当前 Amazon C8i 校准策略为：

$$
p_i=
\begin{cases}
\texttt{m2n2}, & M_i<76,\\
\texttt{m1n4}, & M_i\ge76.
\end{cases}
$$

`m2n2` 的 M1--16 tail 和 `m1n4` 的奇数末尾 N block 由 `m1n2` exact-tail kernel
处理。准备 JIT cache 时 key 同时包含 pattern 与 exact M，因而同一次调用中不同
route 数的 experts 可以安全选择不同 pattern。

令 $K_{13}=\operatorname{round\_up}(H,32)$、
$K_2=\operatorname{round\_up}(F,32)$。两级 loop-order 的单个 packed-B block
都是 $64K_s$ bytes，自动窗口为：

$$
u_{13}=\max\left(1,\left\lfloor
\frac{2^{20}}{64K_{13}}\right\rfloor\right),\qquad
u_2=\max\left(1,\left\lfloor
\frac{2^{19}}{64K_2}\right\rfloor\right).
$$

当 $p_i=\texttt{m1n4}$ 且 $u_s>1$ 时，再把 $u_s$ 向下对齐为偶数，以保持成对
N block。H=4096、F=512 时得到 W13=4、W2=16。零值 override 明确恢复不分窗
loop order，正整数 override 强制 block 数；AVX-512 的默认值仍为零。

AVX-512 与 AMX executor 还共享一个不进入 planner 的确定性 team-N mapper。令
$T\in[1,256]$ 为请求 worker 数，$A$ 为 active expert 数，按 route 数降序记为
$M_{(1)}\ge M_{(2)}\ge\cdots$。W13/W2 可独立拥有的 N 单元数为：

$$
q_{13}=\frac{\operatorname{round\_up}(F,16)}{16},\qquad
q_2=\frac{\operatorname{round\_up}(H,32)}{32},\qquad
q=\max(q_{13},q_2).
$$

一个包含 $t_i$ 个 worker 的 expert team 最多取 $t_i\le q$；第 $\ell$ 个 worker
对任一 stage 的 $q_s$ 个 block 拥有区间：

$$
b_{s,\ell}=\ell\left\lfloor\frac{q_s}{t_i}\right\rfloor
 +\min(\ell,q_s\bmod t_i),\qquad
e_{s,\ell}=b_{s,\ell}+\left\lfloor\frac{q_s}{t_i}\right\rfloor
 +\mathbf 1[\ell<q_s\bmod t_i].
$$

这些区间两两不交并覆盖 $[0,q_s)$。team 内先并行 gather（AVX-512 按 M12
panel，AMX 按 logical row），barrier 后计算 W13 区间，再经第二个 barrier 计算
W2 区间。故 W13 intermediate 和 W2 output 都是 single-writer，最终 token merge
仍保持原来的 TopK 顺序。

对一个 wave 中的 expert 集合，先令所有 $t_i=1$。在
$\sum_i t_i<T$ 且仍有 $t_i<q$ 时，重复选择：

$$
i^*=\arg\max_i\frac{M_i}{t_i},\qquad t_{i^*}\leftarrow t_{i^*}+1.
$$

runtime mapping 的分支顺序为：

1. 若 $M_{(2)}>0$、$M_{(1)}\ge64$ 且
   $M_{(1)}\ge2M_{(2)}$，按 $M_i$ 降序生成 waves。当前 wave 的最大 route 为
   $M_{\max,w}$、剩余 expert 数为 $R_w$ 时，放入的 expert 数为
   $a_w=\min(R_w,\max(1,\min(T,\lfloor64T/M_{\max,w}\rfloor)))$，再按上式
   分配 team width；
2. 否则若 $A<T$，所有 active experts 放入同一个 cooperative wave，再按上式
   分配剩余 workers；
3. 否则保留 atomic expert queue，每个被领取的 expert 使用一个 worker。

64-row 与 2x skew 门槛是 Amazon C8i 的 executor heuristic，不是体系结构常数。
team scratch 按同一 slot 跨 waves 的最大 $M_i$ 分配；merge worker 数取
$\min(T,\text{tokens})$。这一 mapper 改变实际 $I_i(t)$ 和 active configuration
$\mathcal Z$，但不扩大 `IntervalPlanner` 的 width/shape/ordering candidate set。

该 policy 只改变 $I_i$/$D_i$ 的 runtime 实现响应，不改变现有 planner 的 shape、
assignment 或 ordering 候选。M=76 阈值和 1 MiB/512 KiB budget 目前只在 Amazon
C8i、H=4096/F=512 上做过性能校准；team/wave 门槛只在 8-core C8i 上验证。
跨 CPU 或显著不同 model dimension 时必须重新做 held-out route/thread 验证，
不能把这些阈值解释为体系结构常数。

## 9. 剪枝验证

> 本章按时间保留实验记录。凡使用 `R13/R2`、split、stage-window 或旧 profile
> 文件名的结果均为 v0.89 之前的历史证据，不能直接校准当前 full-N runtime；当前
> 验证状态以 9.34 和变更记录 v0.89 为准。

令未剪枝 oracle 的最优值为 $C^*$，求解器下界为 $LB$，剪枝 planner 的可行
结果为 $C_{\mathrm{pruned}}$。则：

$$
0\le
\frac{C_{\mathrm{pruned}}-C^*}{C^*}
\le
\frac{C_{\mathrm{pruned}}-LB}{LB}.
$$

对未证明最优的 CP-SAT 运行，还应同时报告 incumbent $UB$。此时实际 regret
位于
$[\max(0,C_{\mathrm{pruned}}/UB-1),\,C_{\mathrm{pruned}}/LB-1]$；
不能把 `FEASIBLE` incumbent 当成 exact optimum。若 duration 来自实测中位数或
拟合 $\widehat I$，该区间只对性能近似模型成立；只有每个 duration 都是保守的
物理时间下界时，$LB$ 才能解释为硬件级绝对下界。

每一层剪枝都应分别报告：

- plan-space 缩减量；
- planning latency；
- 相对 oracle 的实际 regret；
- oracle 未证明最优时，由 $LB$ 得到的保守 regret 上界；
- 在真实 event simulator/runtime 上的 held-out regret。

性能近似还必须分层验证：$\widehat I$ 报告未见 route/thread 的时间误差和排序
误差；$\widehat D$ 报告未见同构及异构 active configuration 的 slowdown 误差；
最终 planner 报告在真实 runtime 上的 schedule regret。单独证明 isolated table
准确，不能证明 contention-aware planner 准确。

### 9.1 Amazon 192-core NUMA0 扩展校准

2026-07-14 在 Neoverse-V3 NUMA0 的 CPU `0-95` 上生成 TP4、TP2、EP4、EP2
的 split/no-split schema-v2 profile。每个 profile 包含 117 个 isolated 点
（13 个 route、9 个线程宽度）和 120 个 homogeneous contention 点，使用 5 次
warmup、20 次正式采样，并保留原始 samples。所有 profile 通过 schema、kernel
identity、split-pair grid 和 `ContentionCostModel` 加载验证。

测量证明 32T 以上的 isolated scaling 与 expert shape 相关：TP4 split 的短
route 常在 32--64T 饱和，而 EP4 长 route 可继续受益到 96T。后来定位到
64-local-expert TP profile 的“每个 expert 都有 2040 routes”full-call cliff
包含每轮重新分配并首次写约 1 GiB BF16 output 的成本，不能解释为 GEMM
contention。2026-07-16 起 profiler 为每个测量点预分配 native `out`，warmup
完成 first-touch，正式样本只测稳态 expert compute。旧 profile 中缺少
`measurement.output_buffer=preallocated_reused_native_out` 的大总 route 点不得
用于校准 $D_i(\mathcal Z)$；allocation/page-fault 应作为独立 E2E 成本处理。

使用 route 96/384/1536 作为 holdout 时，8 个 profile 的公式中位绝对误差为
0.53%--2.10%；P90 在 `R13=2` profile 上不超过约 7.3%，但 EP/`R13=1` 可达到
约 11%--12%。因此 exact table 与小 M residual 仍是 active profile 的必要部分，
不能仅凭 USL 主公式替代高线程尾部校正。

### 9.2 Exact-range owner-cache 留出验证

Neoverse-V3 NUMA0 有 96 核、每核 2 MiB 8-way L2。EP2 `R13=2,R2=1` stage 为
16 MiB；预留 2 way 后 $C_{\mathrm{owner}}=144$ MiB。独立 owner-slice scan
拟合得到 $B_{\max}=8.991$ TB/s、$n_s=0.85$，取 $u=95\%$ 后预测 band 为
3--8 experts，即 48--128 MiB。resident scan 拟合误差 median 1.1%、max 7.4%。

route 1020/2040 fused holdout 经 $T_{\mathrm{iso}}$ 5% headroom 后均推荐
4 experts / 64 MiB，真实 regret 分别为 0.33% 和 0.00%；128 MiB 仍分别只慢
3.0% 和 1.0%，而 160 MiB 已同时下降约 18%。现有 route 192/768/2040
profile 上推荐 shape 的 regret 为 0.00%/0.79%/1.57%。route 48 regret 为
17.27%，因此短 route 明确保留 empirical contention table，不进入该 band
规则的有效域。

跨机器留出使用 8-core Neoverse-V1、每核 1 MiB 8-way L2 和 TP4 的 4 MiB
split stage。6/8-way owner budget 为 6 MiB，因此公式退化为 1 expert / 4 MiB；
route 1020/2040 实测最优也均为 4 MiB，regret 都是 0.00%。独立 scan 从 4 MiB
的 784 GB/s 降至 16 MiB 的 266 GB/s，说明 32 MiB shared L3 并不能替代
N-split owner-private residency。该交叉验证只支持 split 路径；no-split 不在
本公式和验证范围内。

上述留出只验证 legacy 4/16 MiB-equivalent stage。新增 byte-window variant 的
首轮 NUMA0 验证固定总线程为 96、TP4 shape、每专家 route 2040：`24x4T` 的
4/2/1 MiB 时间为 30.55/32.14/36.73 ms，`48x2T` 为
85.90/61.72/65.79 ms，`96x1T` 为 310.61/179.36/133.12 ms。逆序复测保持同一
排序。三个 schedule 的最优点分别为 4/2/1 MiB，等价于约 1 MiB packed B / active
worker，同时 nominal aggregate window 都约为 96 MiB；因此当前实验尚不能独立
识别 private-L2 与 aggregate-cache 两种作用。

该结果证明 byte window 必须与 $(M,t)$ 联合解释，而不能把 1 MiB 或 2 MiB
设成全局默认值。早期实现曾把 window/range 放入 profile identity 并联合搜索
`(window, shape)`；该设计已在 v0.88 删除。当前由逐 task 的确定函数
$g_\theta(M,t)$ 选择窗口，profile 只提供一个 calibration geometry 和响应表，
不再存在 operator-wide split/no-split 候选。原始数据见
`optimizations/fused_moe_sve/results/amazon_192c_weight_windows.md`。

2026-07-26 先增加了 async benchmark 的 W13/W2 独立窗口覆盖，以检验另一台
机器上观察到的每 worker `W13=1 MiB, W2=0.5 MiB` 是否可迁移。V3 NUMA0
`24x4T` 下，该组合相对两段均为 1 MiB/worker，在 route 192/384 分别快
6.0%/2.2%，route 768 无可分辨差异，但 route 1536/2040 分别慢 2.1%/2.6%；
NUMA1 复现 route 2040 的 2.4% 回退。route 384 的二维扫描得到 W13 约
2 MiB/range、W2 约 0.5--1 MiB/range，而 route 2040 仍以两者 4 MiB/range
最好。因此 stage 最优窗口可以不同，但同时依赖 route 和 team width，不能从
单机常数或固定 W13:W2 比例外推。完整 microbenchmark 数据见
`optimizations/fused_moe_sve/results/amazon_192c_stage_weight_windows.md`。
route 384 的 300-call PMU 对照还显示，`4:4 -> 2:0.5 MiB` 时指令数仅变
-0.06%、L2D refill 反而增加 20.7%，但 LL-cache read miss 减少 6.9%、
memory-stall cycles 减少 39.9%。因此中等 route 的收益不是减少 B load 指令，
而是缩短 reuse distance、降低高延迟 refill 比例；不能只用 L2 refill 数解释。

随后将该覆盖提升为 Plan V2 的可选 per-task 字段，并实现只依赖 $(M_i,t_i)$ 的
机器专用 post-plan policy。192-core 主机 NUMA0 `0-95`、TP4
`H=4096,F=512,E=256,topk=6` 的 21-run 中位数对照保持 expert assignment、
task width、DAG 和 merge 策略完全一致，只改变 stage windows：

| workload | 主要 route / width | auto ms | static ms | gain |
| --- | ---: | ---: | ---: | ---: |
| uniform | 48 / 16T，未覆盖 | 16.518 | 16.572 | -0.32% |
| active-set-32 | 384 / 8T | 8.697 | 8.555 | +1.66% |
| active-set-64 | 192 / 8T | 10.007 | 9.066 | +10.38% |
| active-set-128 | 96 / 8T | 11.932 | 9.312 | +28.14% |
| tiered-hotspot | mixed / 8T | 8.771 | 8.179 | +7.23% |
| long-short-bimodal | 16T + 1T，未覆盖 | 10.411 | 10.419 | -0.08% |
| captured DSV4 | mixed；25 tasks 覆盖 | 14.698 | 14.730 | -0.22% |

未覆盖 case 的最大观测回退为 0.32%，属于本机噪声量级；收益集中在大量
96--384 route、8T expert 同时活跃的情况。captured 分布中只有少数 task 被
覆盖，whole-call 没有可分辨收益，所以该 policy 不能外推为跨机器或跨 shape
默认值。NUMA1 CPU `96-191` 的独立复测得到 active-set-64
`9.955 -> 9.002 ms`（+10.58%）、active-set-128
`11.932 -> 9.199 ms`（+29.71%），与 NUMA0 的收益一致。因此它只在上述精确
profile identity 和两个已验证 rank CPU 集合上默认开启；显式 opt-out 和所有
不匹配 profile 均保持继承行为。测试 extension hash 因新增 Plan V2 ABI 与
profile 不同，但 benchmark 逐字段断言两种 variant 的 plan 除两个 window
数组外完全一致；结果只证明 runtime policy 收益，不构成新 profile 的绝对时间
校准。默认开启后的无 enable-flag 集成复测在 NUMA0/NUMA1 的
active-set-128 分别提升 30.58%/30.34%，NUMA0 未覆盖 uniform 仅变化 0.01%。
完整记录见
`optimizations/fused_moe_sve/results/amazon_192c_static_stage_window_policy.md`。

### 9.3 Ready-token combine 首轮验证

2026-07-16 在 Neoverse-V3 NUMA0 CPU `0-95` 上验证默认 async 候选路径，固定
`tokens=2048`、`top_k=6`、`H=4096`、`F=512`、12 experts、每专家 8T，并在每轮
内随机化 control/candidate 顺序。均衡 route 下负载门槛关闭实验路径，三轮
31-sample median 差异为 -0.14%、+0.10%、-0.29%。25%/75% 两组不相交 TopK
分布下，trace 显示 605/2048 个 token 在 expert 阶段结束前完成 merge；三轮
median 差异为 +0.23%、+0.33%、+0.29%，仍低于约 1% 噪声下界。

首个 per-route 原子减计数原型在均衡分布上从 7.754 ms 退化到 10.116 ms，已被
拒绝。当时实现改为每 expert 一次 completion publication、只读 TopK 状态检查、
每 token 一次 CAS 和按 expert 批量入队。该结果只证明第一版在两个受控分布上
没有可分辨的 median 回退；尚不能证明对 captured routing 有净收益，也不能作为
planner 中 merge-overlap 的 cost 校准。

2026-07-30 在同一机器 NUMA0 CPU `0-95` 上重新验证同轮 queue drain。输入为
`tokens=2048`、`top_k=6`、`H=4096`、`F=512`、256 local experts 的
active-set-8，使用当前 `6x16T -> 4x24T` route-sliced production plan、BF16
direct-route store、同一份 packed weights，并在 101 轮内随机化 variant 顺序。
旧 early-ready 加连续 final merge 的中位时间为 8.691 ms；同轮 drain 的
batch `1/2/4/8` 分别为 `8.659/8.637/8.619/8.648 ms`。默认 batch 2 相对旧路径
提升 0.63%，P90 从 8.963 ms 降到 8.706 ms（2.95%）。轻量 stage counter
显示旧路径在 resident worker job 内只完成 `584/2048` 个 token，余下 1464 个
进入第二轮 merge；新路径为 `2048/2048`，没有 final worker dispatch。

受控 25%/75% two-group 的 101 轮中，旧/新中位数为
`10.266/10.181 ms`（新路径 +0.83%）；均衡输入触发 team-load gate，三种配置
差异不超过 0.15%。batch 1--8 的 spread 只有 0.040 ms，因此当前证据支持
“同轮 drain 消除收尾和重调度”，但不能单独证明软件预取带来可分辨收益；
$B_{\mathrm{merge}}=2$ 是保留一次 lookahead 且避免大 batch 尾部失衡的保守值。

Plan V2 三态控制的 contract 验证要求为：`null/false/true` 必须分别下沉为
`-1/0/1`；经验和解析 DAG simulator 返回的最大 task finish 必须等于原
makespan；相同 expert finish 的 strict bridge 必须输出 `false`，存在 finish
差时必须保持 `null`；显式控制连接旧 native extension 时必须报错。AArch64
集成验证还需对 auto/off/on 做 bit-exact 对比，并用
stage timing 确认 off 路径没有 ready-token publication/owner scan 且启动统一连续
merge。fixed-owner 路径的 trace 还必须逐 token 验证
$q\in\mathcal Q_{\mathrm{tid}(q)}$，并确认 drain 模式恰好产生 $|\mathcal Q|$
次 merge、没有 final merge fallback。

2026-07-31 在 AmazonC5192Cores NUMA0 `0--95` 上完成 fixed-owner 首轮验证。
`tokens=2048, top_k=6, H=4096, F=512, E=12` 的 25%/75% two-group 分布、
96T、`R13=2,R2=1`、7 次 warmup 和 31 次交错采样中，post-barrier、fixed-owner
early+final、fixed-owner same-job drain 的中位数分别为
`10.339/10.279/10.216 ms`；drain 相对 post-barrier 提升 `1.20%`，P10/P90 为
`10.177/10.537 ms`。该组只验证新 owner policy 相对同轮 fallback 没有性能回退，
不作为与旧 global queue 的跨版本对比。AArch64 完整 MoE 测试为
`102 passed, 3 skipped`，trace 对每个 merge record 验证 token 落在执行 tid 的
静态连续 owner 区间内。

### 9.4 Amazon 192-core 双 NUMA 稳态校准

2026-07-16 在 Neoverse-V3 的全部 192 核上以两个同步 rank 重测 TP4/F512
`R13=2,R2=1` profile：rank 0 使用 CPU `0-95`/NUMA0，rank 1 使用 CPU
`96-191`/NUMA1，每轮取两个 rank wall time 的最大值。每个 rank 流式使用 64 份
不同 expert 权重；profile 包含 117 个 isolated 点和 120 个 contention 点，使用
5 次 warmup、20 次正式采样，并复用预分配 native `out`。

稳态结果消除了旧 allocate-per-call cliff。`64 experts x 2040 routes` 的 `[96]`
shape 从旧口径约 5.53 s 降为 250.88 ms，derate 为 0.988；全表最优是每 rank
`24x4T`，wall time 91.54 ms，双 rank 聚合吞吐 35.89 TFLOP/s。代表性最优点为：

| routes/expert | 每 rank shape | 双 rank wall | 聚合吞吐 |
| ---: | --- | ---: | ---: |
| 1 | `12x8T` | 2.03 ms | 0.79 TFLOP/s |
| 12 | `24x4T` | 2.12 ms | 9.10 TFLOP/s |
| 48 | `6x16T` | 3.98 ms | 19.43 TFLOP/s |
| 192 | `6x16T` | 9.34 ms | 33.12 TFLOP/s |
| 768 | `6x16T` | 34.14 ms | 36.24 TFLOP/s |
| 2040 | `24x4T` | 91.54 ms | 35.89 TFLOP/s |

M=768 时 `12x8T` 与 `6x16T` 相差约 0.1%；M=2040 时更细的 `24x4T`
重新占优，等价于每 rank 同时保持约 96 MiB split packed-B stage。两个 NUMA 的
isolated median 在长 route 上接近：M=2040 的 32/64/96T 最大比值分别为
1.056/1.017/1.034；短 route 的 64/96T 可出现更大不对称，因此 profile 必须保留
pairwise-max 聚合，不能用两个 rank median 的平均值替代。

### 9.5 256-expert TP4 双 NUMA 校准

2026-07-17 按真实 256 routed-expert TP4 人口重测同一 F512
`R13=2,R2=1` kernel。TP 不分割 expert 数量，因此每个 rank 都生成并轮换
256 份不同权重；rank 0/1 仍分别绑定 CPU `0-95`/`96-191` 和
NUMA0/1。profile 包含 117 个 isolated 点和 140 个 contention 点，
每点 5 次 warmup、20 次正式采样，并按逐轮两 rank 的较慢者聚合。

代表性最优点为：

| routes/expert | 每 rank shape | 双 rank wall |
| ---: | --- | ---: |
| 1 | `48x2T` | 8.67 ms |
| 12 | `24x4T` | 9.04 ms |
| 48 | `6x16T` | 16.02 ms |
| 192 | `6x16T` | 36.04 ms |
| 768 | `12x8T` | 125.88 ms |
| 2040 | `24x4T` | 329.79 ms |

M=12 存在宽平台：`24x4T`/`48x2T`/`96x1T` 分别为
9.042/9.049/9.094 ms，相对最优只相差 0.0%/0.1%/0.6%。因此
96 个单线程 expert 并不差；旧 E64 profile 未包含该可行 shape，
其 `24x4T` 结论只适用于 64-task full call，不得作为 256-expert
TP4 planner 的总体调度结论。本轮只替换 profile identity 和验证数据；
$I_i(t)$、$D_i(\mathcal Z)$ 定义及 production 剪枝不变。

### 9.6 M12 冷 B 带宽扩展曲线

2026-07-17 在 Neoverse-V3 NUMA0 CPU `0-95` 上固定 TP4/F512、M=12
和每 expert 1T，扫描 1--96 个同时 active experts。每个样本只执行一个
完整并发波，样本间在 256 份独立 packed weights 上循环移动窗口；
因此不包含固定总任务数产生的 partial-wave tail。点顺序随机化，每点
5 次 warmup、30 次正式采样。

M=12 只有一个物理 M panel，每个 expert 的 W13+W2 packed B 合计
12 MiB 且无 panel 间复用。定义冷 B 有效带宽为：

$$
B_{B,\mathrm{eff}}(n)
=\frac{n\cdot 12\ \mathrm{MiB}}{T_{\mathrm{one\ wave}}(n)}.
$$

该值包含 GEMM 的 load issue、BFMMLA 与 dispatch 重叠，是 planner 可用的
effective packed-B rate，不是 PMU memory-controller byte count。代表点为：

| 1T experts / threads | wall | $B_{B,\mathrm{eff}}$ | 相对 96T |
| ---: | ---: | ---: | ---: |
| 1 | 0.559 ms | 22.5 GB/s | 6.4% |
| 8 | 0.812 ms | 124.0 GB/s | 35.2% |
| 12 | 0.831 ms | 181.7 GB/s | 51.6% |
| 24 | 1.136 ms | 265.9 GB/s | 75.5% |
| 48 | 2.021 ms | 298.8 GB/s | 84.8% |
| 64 | 2.514 ms | 320.4 GB/s | 91.0% |
| 86 | 3.219 ms | 336.2 GB/s | 95.5% |
| 96 | 3.430 ms | 352.1 GB/s | 100.0% |

按“从该线程数开始，后续所有实测点都不低于目标”的稳健口径，
50%/75%/90%/95% 峰值带宽分别需要 12/24/64/86 个 1T experts。
该曲线可作为未来 DRAM pressure shadow 的 NUMA-local service response，但当前
planner 仍使用 empirical contention profile；公式、可行域和 production 剪枝未改变。

### 9.7 Xbyak exact-M kernel 验证

2026-07-20 在 Neoverse-V3 NUMA0 CPU `0-95` 和 8-core Neoverse-V1 CPU
`0-7` 上，以 H4096/F512、8 份连续 expert、`R13=2,R2=1` 对比
`jit/xbyak_exact_m` 与 `asm/static_bucketed`。每次实现切换后先执行一次不计时
调用，再连续测 5 次，避免把 instruction-cache 切换计入某一 variant；每点取
11 次样本中位数。M1--M12、SiLU poly4/5/6、normal/scheduled/async 和 W2
direct-route 均与 static asm bit exact。

V3 单线程 M5/M6 分别提升 11.75%/11.75%，M9/M10 提升
10.07%/10.44%；M12 回退 0.46%。M192/M2040 在 1--96T 的全部 control 点落在
`[-0.86%, +0.79%]`。V1 的 M5/M6/M9/M10 在 1--8T 提升约 5%--12%，M12 与
长 route control 最差回退 0.82%。因此 exact-M JIT 进入 production 默认，static
asm 保留 fallback；但该结论只证明 kernel variant 本身，不允许继续复用旧
bucketed-M 的 isolated/contention 表。新 profile 必须覆盖 route 1--12，并记录
`sve_implementation=jit`、`m_tail_policy=xbyak_exact_m` 及 Xbyak commit。

同日生成两组完整 split/no-split schema-v2 pair。8-core V1 standalone
F512/E8 每个 policy 含 80 个 isolated 点和 68 个 contention 点；192-core V3
双 NUMA TP4 F512/E256 每个 policy 含 180 个 isolated 点和 238 个 contention
点，双 rank 样本按 pairwise maximum 合并。8-core 的 M192/M2040 上 split
吞吐分别提升 5.34%/4.25%，但双 NUMA E256 的同两点分别回退 3.03%/2.72%；
这直接否定跨机器或跨 workload 固定选择 split 的支配假设。两组 profile 的
route grid 均直接覆盖 1--12，split pair 的 source/binary hash、Xbyak commit、
线程和 shape 网格一致，并由 catalog 回归测试验证。

### 9.8 调度论文输入分布

固定的 synthetic validation catalog 使用 $T=2048$、$K=6$、$E=256$，因此
$\sum_iM_i=12288$。它覆盖四类互补输入：

| Family | Histogram | 控制变量 |
| --- | --- | --- |
| uniform | `256x48` | 理论均匀基线 |
| active-set sweep | `8x1536`、`16x768`、`32x384`、`64x192`、`128x96`、`256x48` | 固定 routes，改变活跃权重工作集和单 expert route |
| tiered hotspot | `4x768 + 12x384 + 48x96` | hot/warm/cold 三层负载 |
| long-short bimodal | `5x2040 + 174x12` | 长 route compute 与 M12 冷权重流量并存 |

所有 synthetic histogram 都满足 2.1 节的全局 TopK 约束。active-set 的 256
endpoint 与 uniform 共用 `moe256-uniform`，避免重复 workload。它们只用于
planner/cost-model 的受控比较；真实有效性仍由 captured routing trace 验证，
不得把 synthetic preset 当作真实 router 概率模型。此次扩展不改变
$I_i(t)$、$D_i(\mathcal Z)$、目标函数、线程宽度集合或 production 剪枝。

### 9.9 单核 packed-B service ceiling 与 exact-M 效率

2026-07-23 在 Neoverse-V3 NUMA0 CPU 48 上补充 H4096/F512、单 expert、
单线程的冷权重校准。standalone reader 只执行 8-way-unrolled SVE `LD1H`，
不执行 BFMMLA、epilogue 或输出写回；768 MiB 连续读取的两轮中位数为
41.250/41.223 GB/s，64 份 12 MiB chunk 乱序轮换为 40.051/40.035 GB/s。
后者与每个 expert 的 W13 8 MiB + W2 4 MiB packed-B 粒度一致，因此定义本机
单核 production-like service ceiling：

$$
\beta_{B,1}^{\mathrm{V3}}=40.04\ \mathrm{GB/s}.
$$

该值是 CPU 侧 effective packed-B read ceiling，不是 uncore PMU 的 DRAM
controller byte rate。单核 kernel 的访存效率统一定义为

$$
\eta_{\mathrm{mem}}(M)
=\frac{6HF/T_{\mathrm{W13+W2}}(M)}
       {\beta_{B,1}^{\mathrm{V3}}},
$$

其中分子按每个 expert 12 MiB packed weights 只扫描一次计算。计算侧同时保留
三个不可互换的量：

$$
m_c=2\left\lceil\frac M2\right\rceil,\qquad
\eta_{\mathrm{lane}}=\frac M{m_c},
$$

$$
\eta_{\mathrm{compute,useful}}
=\frac{6MHF/T_{\mathrm{W13+W2}}}{403.8\ \mathrm{GFLOP/s}},\qquad
\eta_{\mathrm{issue}}
=\frac{6m_cHF/T_{\mathrm{W13+W2}}}{403.8\ \mathrm{GFLOP/s}}.
$$

64 份不同 expert 权重轮换、每 M 192 个样本的代表点为：

| M | GEMM stage | $\eta_{\mathrm{lane}}$ | useful GFLOP/s | $\eta_{\mathrm{compute,useful}}$ | $\eta_{\mathrm{issue}}$ | packed-B GB/s | $\eta_{\mathrm{mem}}$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3472 ms | 50.0% | 36.24 | 8.97% | 17.95% | 36.239 | 90.51% |
| 2 | 0.3447 ms | 100% | 73.00 | 18.08% | 18.08% | 36.500 | 91.16% |
| 4 | 0.3580 ms | 100% | 140.58 | 34.81% | 34.81% | 35.144 | 87.77% |
| 8 | 0.4336 ms | 100% | 232.16 | 57.49% | 57.49% | 29.020 | 72.48% |
| 12 | 0.5245 ms | 100% | 287.90 | 71.30% | 71.30% | 23.992 | 59.92% |

完整 M1--M12 表与复现命令位于
`optimizations/fused_moe_sve/results/amazon_192c_single_core_m1_m12_efficiency.md`。
相邻 odd/even M 具有相同物理 row-pair 数，因此 stage time 与
$\eta_{\mathrm{issue}}$ 基本相同；odd M 的差异由
$\eta_{\mathrm{lane}}$ 单独表达。40.04 GB/s ceiling 只作为后续解析模型的
机器校准常数；本轮不修改 active planner、$I_i(t)$、$D_i(\mathcal Z)$、
contention table 或 production 剪枝。

### 9.10 x86 AMX 自动子变体验证

2026-07-20 在 2-core Amazon C8i（Intel Xeon 6975P-C）上验证 8.4 的确定性
runtime policy。H=4096/F=512 的 m2n2/m1n4 AB/BA 配对长采样显示 M=68/72
位于约 ±2% 的尾部/频率噪声区；M=76 时 `m1n4` 单核为 1.207 vs 1.253 ms、
双核为 1.522 vs 1.536 ms，并在 M=80 以后继续领先。因此当前保守阈值取 M=76。
skewed route histogram `[384,19,19,18,18,18,18,18]` 下，per-expert auto
混合策略单核为 9.036 ms，比最佳全局固定 pattern 的 9.427 ms 快 4.3%；双核
为 10.595 vs 10.665 ms。

自动 byte window 在 M=16 相对显式 0/0 不分窗无可分辨回退；M=48/80/256/2048
的单核 speedup 为 1.35x/1.94x/2.31x/2.09x，双核为
1.21x/1.53x/1.81x/1.65x。H=4096/F=512 的公式值 4/16 与无环境变量路径在
单核扫描中相差不超过 0.4%。focused x86 correctness suite 为 95 passed、4 skipped，
并覆盖同一次调用内 M=75/M=76 两种 pattern、H/F/K/N tails、双线程以及 AMX
kill-switch 回退 AVX-512。完整命令、样本和热状态限制见
`optimizations/fused_moe_avx512/results/amazon_c8i_2core_auto_dispatch_20260720.md`。

该验证只校准 x86 runtime 的确定性响应 mapper，不改变 CPU MoE planner 的决策
变量或 candidate space，因此本次不修改 planner/cost-model 公式测试。跨 CPU 或
显著不同 H/F 的 policy 仍需单独 held-out 验证并更新 profile identity。

### 9.11 x86 cooperative N-split 与 skew-wave 验证

2026-07-22 在 8-core Amazon C8i（Intel Xeon 6975P-C，8 physical cores、1 thread
per core）上验证 8.4 的 team-N mapper。固定 H=4096、F=512、BF16、top-k=1，
使用 CPU `0-7`、`OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=close`、
`OMP_PLACES=cores`、`OMP_WAIT_POLICY=PASSIVE`、8 次 warmup 和 31 次正式采样；
权重 prepack/JIT warmup 不计入运行时间。AMX auto pattern 的 median 如下（ms）：

| route histogram | 1T | 2T | 4T | 8T | 1T/8T speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[2048]` | 26.446 | 14.209 | 7.917 | 5.755 | 4.60x |
| `[1536,256,256]` | 39.225 | 23.311 | 14.587 | 11.893 | 3.30x |

同一进程配置下，PyTorch/oneDNN staged baseline 的 8T median 分别为 19.611 ms
与 23.640 ms，因此 fused executor 分别快 3.41x 与 1.99x。M=256 单热 expert
从 1T 3.574 ms 降到 8T 0.855 ms（4.18x）。AVX-512 BF16 单热 M=2048 也从
262.063 ms 降到 39.461 ms（6.64x），证明 N ownership 对两种 x86 backend 都
生效；但该 AVX-512 kernel 仍慢于同线程数 oneDNN，不能把 scaling 误写为绝对
kernel 优势。

另一个 12-warmup/51-sample 小 M sweep 中，单热 M=1/4/16/48/64 的 8T 相对 1T
分别为 7.49x/2.76x/4.89x/3.69x/3.99x；在该 H/F 与机器上没有观察到需要按 M
进一步压低 team width 的性能边界。

为避免 cooperative barrier 在本来已有足够 expert parallelism 时造成回退，最终
policy 不对均衡 route 强制 wave。E2 `[1024,1024]` 与 E8 `[256,...,256]` 的 8T
median 分别为 9.183 ms 和 8.709 ms，仍走原 expert queue；对应 oneDNN 为
21.747 ms 和 23.882 ms。仅在 2x/64-row 强偏斜门槛命中时才使用有序 waves。

x86 correctness suite 为 163 passed；测试覆盖 AVX-512 和 AMX
`m1n2/m2n2/m1n4`、1/2/4/8 threads、H/F/K/N tails、单 expert、active expert
不足、均衡 8 expert、active expert 数等于线程数的强偏斜 wave，以及 worker
异常时 barrier cancellation。1T 与 8T 输出要求 BF16 bit-exact，并同时对 naive
reference 做容差比较。完整命令和原始 median/best 数据见
`optimizations/fused_moe_avx512/results/amazon_c8i_8core_nsplit_20260722.md`。

本次变化是 synchronous x86 executor 的确定性 response mapper；没有修改
planner candidate space、profile schema 或 cost-model 参数，所以不新增
planner/cost-model 公式测试。若将该 team/wave mapper 提升为 planner 可选动作，
必须先为不同 $t_i$/wave active set 重建 $I_i(t)$、$D_i(\mathcal Z)$ profile 并做
held-out regret 验证。

### 9.12 Plan V2 native strict/tail-pool bridge 验证

`IntervalPlanner` 默认对每个已选 task 生成 singleton allowed-width CSR、
whole-expert stage、空 resize mask、fixed placement 和 logical-core interval；
cache hit 重建路径生成相同 bridge。`AsyncMoEPlanV2` 和 ARM native entrypoint
独立检查版本、CPU 映射、task 数组长度、依赖 DAG、允许宽度 CSR、
selected/preferred/min/max、placement，以及当前 stage/range/resize 子集。
strict V2 直接进入同一个 native async executor，不读取 legacy short-pool
环境变量；旧扩展只可通过 Python adapter 执行 strict 降级。

tail-pool bridge 将 `routes <= max_pooled_routes` 的 whole-expert task 标为
pooled，使用 `core_begin=-1` 和统一 pool width。它删除 pooled task 的前驱，并
把后续 fixed task 的依赖穿过连续 pooled 链重连到最近 fixed ancestors。Python
与 native 同时验证 pool width 整除总线程、fixed interval 对齐、pooled task
无依赖、fixed task 不依赖 pooled task。运行时每个对齐 group 等覆盖它的全部
fixed blockers 完成后，从全局队列领取完整 pooled task。

回归覆盖位于 `tests/test_moe_plan_v2.py` 和
`tests/test_moe_cost_model_v2.py`，包括 legacy bridge 升级、strict/tail
placement、较宽但不执行的 width envelope、非法 resize/width/placement/
dependency/alignment 拒绝、native V2 参数传递、strict 旧扩展 fallback，以及
首次/缓存 planner 输出。production planner 默认比较 strict 与自动 tail-pool
候选；`dynamic_tail_pool=False` 保留严格基线，`tail_pool_threads` 保留显式强制
路径。自动候选按 2.3 的 list-scheduled surrogate DAG 使用同一个
contention-aware cost model 评分；缓存 identity 同时包含 auto/strict/forced
模式、forced width、threshold 上限，以及每个候选 threshold 下的 eligible
expert 数。route bucket histogram 仍用于近似相似 workload，但跨越 pooled
eligibility 边界时必须重新规划，不能复用旧 dynamic shape。

`tests/test_moe_native_interval_planner.py` 另外逐项比较 Python 与 C++ 的
table/formula $T_{\mathrm{iso}}$、混合依赖 DAG makespan、strict/auto/forced
候选、bridge 和完整 ranking，并验证 1T/4T cold solver 输出相同。native
planner 是求解器实现替换，不是新的 runtime execution mode。

自动策略扩大了 planner candidate space，但没有证明 regret 收敛；它仍需用新的
机器、shape 和路由分布继续做 held-out 验证。当前自动搜索只扩展 strict 最快
uncertainty band 与至少两个最快 head shape，并对 workload 内实际出现的
`1/2/4/8/12` route threshold 去重，以控制冷搜索成本。

2026-07-26 在 AmazonC5192Cores NUMA0 `0-95` 上使用 `R13=2,R2=1`、
H4096/F512、256 experts、2048 tokens、TopK=6 验证默认自动策略。远端 Plan V2/
planner 聚焦回归为 50 passed，native tail-pool 专项为 1 passed；本地全部 MoE
回归为 1097 passed、176 skipped。所有进入计时的
strict/auto/forced-4T/vLLM 输出均 BF16 bit-exact。

在 `5xM2040 + 174xM12` long-short bimodal 上，planner 自动选择
`6x16T` fixed head、`M<=12`、1T tail pool；21 次交错测量中 strict/auto/
forced-4T/vLLM median 分别为 12.529/10.408/10.801/11.728 ms。auto 相对
strict 提升 20.39%，相对 forced-4T 提升 3.78%。uniform、active-set
8/16/32/64/128、tiered-hotspot 和捕获的 DSV4 路由均保留 strict；相同 native
plan 的 strict/auto 中位数最大差异为 0.38%。这既验证 planner 能启用有收益的
dynamic action，也验证它不会默认把其施加到其余八个 workload。

同一主机 `0-95` 核上的 native cold-search 验证使用
`bench_native_cold_planner.py`，对象构造一次、每次重新执行完整候选搜索和 bridge
生成，5 次 warmup、21 次正式采样。长短双峰的 Python/C++-1T/C++-8T 中位数为
59.784/2.610/1.421 ms，即 22.90x/42.07x；captured routing 为
274.342/11.627/3.488 ms，即 23.60x/78.65x。两者 16T 均比 8T 略慢，因此默认
worker 上限为 8；显式参数仍可覆盖。全部后端选择相同 mode、shape、task、
bridge 和 ranking，makespan 只存在小于 $10^{-12}$ 相对误差的浮点求和差异。

此前相同路径的 cache hit 为 bimodal 0.944 ms、真实路由 2.396 ms；cache 仍可
避免 cold search，但 native 后首次搜索已不再是几十到几百毫秒的 Python 瓶颈。
运行扩展 hash 与校准表仍不一致，所以这些结果只验证求解器等价性和 planning
latency，不用于证明 cost model 的绝对时间或 runtime regret 准确性。

### 9.13 192-core 双 NUMA 当前二进制校准刷新

2026-07-26 在 AmazonC5192Cores 上按 9.5 的 256-expert TP4/F512 口径重测
`R13=2,R2=1` 和 `R13=1,R2=1` 完整配对表。两个同步 rank 分别绑定
CPU `0-95`/`96-191` 和 NUMA0/1；每张表包含 180 个 isolated 点和 238 个
contention 点，每点 5 次 warmup、20 次正式采样。route/thread/shape grid
保持不变，source SHA 更新为
`a62e0d9425381750fdc859ed97e2a4d1cc33727ffe36dbb75c453f061a924dfb`，
extension SHA 更新为
`e652d9aad3025a4836d0406110bcbdf3fdc1356aaba2bf789595359a58a92559`。

相对 2026-07-20 表，isolated 全点中位变化为 split +1.67%、no-split +2.99%；
homogeneous full-call 全点中位变化为 +12.51%/+12.40%。后一个统计由超宽
shape 明显拉高：M=2040 split `24x4T` 仅从 332.45 ms 变为 334.60 ms，
no-split `12x8T` 从 323.43 ms 变为 329.60 ms，而单个 96T team 分别增加
20.33%/33.27%。新 split 表的 30-run 缩减复测在 18 个重叠 contention 点上
相对完整表中位差 +0.47%，范围 -4.13% 到 +1.92%，确认长 route 趋势可重复。

在全部 17 个 homogeneous route 上继续使用旧跨 policy 最优点，新表 regret
中位为 3.01%、最大为 7.63%。当时 `PolicyAwarePlanner` 的九个默认论文/真实
workload 中，只有 active-set-32 和 tiered-hotspot 改变 policy/shape，旧选择在
新表上的预测 regret 分别为 4.24% 和 2.04%，其余为 0%。因此本次只原子替换
完整 JIT exact-M split/no-split profile pair，不改变 $\widehat I_i(t)$、
$\widehat D_i(\mathcal Z)$ 定义、公式形式、candidate space 或 production
剪枝。完整命令、逐项差异和噪声分析见
`optimizations/fused_moe_sve/results/amazon_192c_cost_profile_refresh_20260726.md`。

### 9.14 解析 backend 的实现与验收状态

2026-07-26 增加 `AnalyticMoeCostModel`、machine calibration schema 和
`validate_analytic_model.py`。纯逻辑测试覆盖：

- 两锚点 service curve 的单核、插值和饱和值；
- exact-M1 按物理 M2 FLOPs 收费；
- 冷 weight 是 compulsory DRAM，而 packed-A refill 是 cache/spillable traffic；
- M<=12 的 one-pass B 不占 reusable LLC budget，但仍占 DRAM service；
- 非整除 N range 不丢 tile，并按尾 range 的实际 active owners 收费；
- 增加 W13 range 数不改变 GEMM executed work，只增加 range cost；
- 多个 phase 对同一资源的 aggregate request 超过 ceiling 时产生 derate；
- 无 measured shape table 的 analytic model 可直接进入
  `IntervalPlanner`/`PlannedMoE`；
- holdout 报告同时给出 isolated/full-call 绝对误差和真实 shape regret。

2026-08-01 将解析 backend 升级为 phase-aware shared-resource v3。每个 N range 从单一平均 phase
改为 `range_setup -> cold_b -> steady_b`，M<=12 只生成 cold phase；测试逐项验证
cold/steady 的 FLOPs、L1、L2、LLC、compulsory/spillable DRAM 求和保持完整 kernel
demand。并发容量改为按资源的实际 requester threads 计算，setup 不再虚增可用
带宽；新增 `active_resource_pressure()` 与 `explain_dag()`，可直接审计每个事件的
物理路径、offered/allocated rate、capacity、utilization 和 dilation。endpoint
load probe 按最大时间下界组合；packed-B L2 reuse 使用独立 repeated-scan 的三点
miss 校准。59 个 focused tests 通过，但仍只证明公式、守恒关系和 planner contract。

这些测试只验证公式不变量和 planner contract，不证明目标机器精度。真实校准表
不得从现有 route/thread cost table 反推；它必须来自独立 matrix、L1、L2、
LLC、DRAM、frontend/epilogue probe 和固定开销测量。旧 schema-v2 表仅作为
holdout oracle。production 切换门槛暂定为 isolated MAPE 不超过 10%、
contention P90 绝对误差不超过 15%、所有验证 route 的最大 measured shape
regret 不超过 5%。在完成 8-core 与 192-core 至少各一份 unseen
route/thread/mixed-distribution 验证前，解析 backend 保持 opt-in。

同日在 AmazonC5192Cores NUMA0 `0-95` 上完成首份独立薄校准。matrix/L1/L2/LLC/
DRAM probe 不读取 routed-expert 表；DRAM sustainable ceiling 为 `395.9 GB/s`，matrix
register-only ceiling 为 `0.413/39.007 TFLOP/s`（1T/96T）。另用既有 packed-B
repeated-scan PMU 数据给出低窗口/2 MiB/4 MiB miss 锚点 `18.0/62.3/86.9%`。
只用 12 个 isolated 点拟合 `51.912 us` expert fixed、`479.10 ns/route` 和共同
stage scale `1.18546`，所有 54 个 contention 点均留出。

108 个 true isolated holdout 的 MAPE 为 `10.22%`；contention P90 绝对误差为
`47.79%`；六个 uniform route 的平均/最大 measured shape regret 为
`3.33/8.17%`。相对 service-only first pass，最大 regret 从 `32.31%` 明显下降，
但三项仍未同时通过 `10/15/5%` gate。主要残差是长 route `1T/2T` 的 spill 过估，
以及中等 route 对 `8T` 相对 `16T` 的 active-window 代价低估；NUMA-wide LLC
power curve 本身还有 `22.9%` MAPE、`59.5%` 最大误差。因此 production 继续使用
empirical backend。完整数据见
`optimizations/fused_moe_sve/results/amazon_192c_analytic_thin_calibration_20260801.md`。

### 9.15 跨 rank lifetime 状态转换

此前 TP/EP evaluator 分别规划各 rank，并以

$$
\widehat C_{\mathrm{old}}=\max_q \widehat C_q^{(P)}
$$

作为 compute wall time，其中 $\widehat C_q^{(P)}$ 全程使用
`concurrent_ranks=P` profile。对于负载不均衡的 EP，这会在短 rank 已完成后
继续向长 rank 收取跨 rank contention，因而系统性高估尾部。

当前实现保留每个 rank 已选定的 task DAG 和 kernel policy。设按
concurrent-rank 模型预测的倒数第二个完成时刻为 $\tau_s$。在
$0\leq\tau<\tau_s$ 时，各 rank 仍按 $\widehat D^{(P)}$ 推进；在
$\tau_s$ 以后只剩一个 rank 时，长 rank 改用同一 machine/kernel/grid identity
的 `concurrent_ranks=1` companion：

$$
\widehat D_q(\tau)=
\begin{cases}
\widehat D_q^{(P)}(\mathcal Z(\tau)), & \tau<\tau_s,\\
\widehat D_q^{(1)}(\mathcal Z_q(\tau)), & \tau\geq\tau_s.
\end{cases}
$$

转换发生在 phase event simulator 内。若 task $i$ 正在 phase $h$，切换前
剩余基准工作为 $r_{ih}^{(P)}$，该 phase 的总基准时间为
$I_{ih}^{(P)}$，则 single-rank 状态初始化为

$$
r_{ih}^{(1)}
=
\frac{r_{ih}^{(P)}}{I_{ih}^{(P)}}I_{ih}^{(1)}.
$$

已完成 phase、DAG dependency 和当前 phase index 均保持不变，因此不会重放
已完成工作，也不会在切换点重复收取 call setup。companion 必须拥有相同的
route/thread grid、kernel policy 和 phase geometry；planner candidate、LPT
assignment、shape pruning 与 runtime plan 均不改变。缺少 companion 时继续
使用旧的全程 concurrent-rank 上界，避免跨 profile 外推。

2026-07-27 在 AmazonC5192Cores 上以 CPU `0-31`/`96-127` 生成当前 JIT
exact-M EP2 H4096/F2048/E32 split/no-split single/dual 配对表。tiered-hotspot
的两个 rank 分别选择 `(32,)` 与 `(16,16)`；旧全程 dual-rank 估计为
55.605 ms，phase lifetime 转换后为 54.770 ms，减少 0.835 ms（1.50%）。
同日 TP4/F512/E256 companion 的 route-48/route-192 构造检查减少
1.939 ms（4.99%）。两组 profile 均通过 source/extension hash、isolated
grid、shape grid 和 phase geometry 一致性检查。

### 9.16 独立 W13/W2 Plan V2 对照

实验验证必须固定 packed weights、SVE GEMM kernel、per-task stage-window
policy、direct route store 和 merge，只改变 stage coupling。至少比较：

1. production whole-expert Plan V2 auto；
2. 全局两阶段且 W13/W2 强制使用同一 plan；
3. 全局两阶段且 W13/W2 独立搜索 strict/tail-pool；
4. 旧 vLLM global N-range task pool。

正确性覆盖 strict/strict、tail-pool/tail-pool 和两个 stage 使用不同线程宽度
的组合，并要求与 production 输出逐元素一致到现有 BF16 容差。性能验证使用
AmazonC5192Cores NUMA0 CPU `0-95`、256 local experts、TopK=6 和默认论文输入
集；同时记录 route/setup、W13、global barrier、W2、merge、E2E、cold planning
time、预测值及实测值。只有 independent staged 在多个未参与校准的 workload
上稳定优于 matched staged，才说明 stage-specific planning 有效；只有它再
稳定优于 whole-expert production，才有证据解除 production stage-coupling
剪枝。第一版 empirical stage 分解尚未通过这些测试前，实验入口保持非默认。

2026-07-29 在该口径下完成 NUMA0 CPU `0-95` 的 9 个 workload 验证。关闭
ready-token merge 后，matched global staging 在 9/9 case 慢于 whole-expert，
中位差 5.23%，范围 2.08%--11.77%；independent staging 也在 9/9 case
落后，中位差 4.23%，范围 2.29%--6.26%。但在 W13/W2 实际选择不同计划的
active-set-128、long/short bimodal 和 uniform 中，independent 相对 matched
分别提升 4.24%、3.28% 和 7.52%。因此 stage-specific planning 的收益成立，
但不足以补偿全局 barrier 和 expert pipeline/cache lifetime 损失。首版 stage
surrogate 对 independent E2E 的绝对时间 MAPE 为 12.8%、最大误差 34.1%，
不能进入 production。完整方法、stage timing 和复测数据见
`optimizations/fused_moe_sve/results/amazon_192c_planned_two_stage.md`。

### 9.17 W13/W2 边界非阻塞伸缩（已退役）

以下内容是历史验证记录。当前源码不提供 elastic execution mode、resize timeout、
preferred cohort 或统计接口；复现需恢复 Git `0b58091`。

实验入口固定使用 whole-expert SVE fused kernel、direct route store、相同 packed
weights 和相同 merge，只改变 W2 的 team width。验证分四层：

1. `timeout=0` 对每个 resizable task 记录首次尝试、自然取得 preferred cohort、
   原 team 回退和 borrowed thread 数，得到
   $P_{\mathrm{natural}}=N_{\mathrm{natural}}/N_{\mathrm{eligible}}$；
2. 对同一 workload 加入有限 timeout，分别验证 `2x8T->16T` 和
   `4x2T->8T`，记录 waited preferred、timeout fallback、总/最大等待时间；
3. 所有 variant 与 strict Plan V2 做逐元素 BF16 输出比较，并报告 E2E median、
   P10/P90 和 aggregate FLOP/s；
4. affinity 横跨 NUMA 时必须在 plan materialization/native validation 阶段拒绝，
   不允许运行时静默缩小或跨节点借线程。

`elastic_stats_out` 的固定字段为 eligible、preferred、fallback、natural、
waited-preferred、timeout-fallback、cohort jobs、borrowed threads、total wait ns
和 max wait ns。ready-token merge 在该实验入口中关闭，避免它消费同一批 idle
workers 并污染“自然重组机会”的定义。

192-core 主机 NUMA0 的实测结论如下：

- `2x8T->16T` 的 `timeout=0` 自然机会率在 active-set-8、tiered hotspot 和
  DSV4 captured 2K/TopK6 上分别为 `0.00% / 0.67% / 6.53%`；
- tiered hotspot 的 `5 us / 20 us` 等待把 preferred assignment 提高到
  `23.44% / 63.84%`，但 E2E 仍比 strict 慢 `9.21% / 9.67%`；
- `4x2T->8T` 的 256 个 `M=1` case 自然机会率为 `0.21%`，E2E 慢
  `23.30%`；
- 零等待路径在 8 个长 expert 上只慢 `2.53%`，但在 64/223/256 个 task
  上的 boundary handoff 开销已不可忽略。

因此第一版 elastic 已完成正确性和机会率验证，但不进入 cost-model 评分或
production 默认候选。完整命令、P10/P90 和原始 JSON 见
`optimizations/fused_moe_sve/results/amazon_192c_w2_boundary_elastic.md`。

2026-07-29 增加 planner 显式 W2 target 后，验证口径扩展为 paired tail：
两个 `16T` source task 可分别指定 `0-31` 和 `32-63` 的 `32T` target。正确性
测试必须强制覆盖一个包含式扩容和一个不相交迁移，并要求两个 preferred
assignment 都发生；性能测试固定比较 strict、零等待和有限等待，统计迁移后
释放 source 所形成的第二次同-pass cohort assignment。

AmazonC5192Cores NUMA0 active-set-8、每 expert `M=1536` 的 strict plan 为六条
`16T` lane，其中 task 1/3 是 `0-15` 和 `16-31` 上的第二轮尾 task。显式指定
task 1 `0-15 -> 0-31`、task 3 `16-31 -> 32-63` 后：

- target-machine SVE 测试中，一个包含式扩容和一个不相交迁移均实际发生，输出与
  strict 逐元素一致；
- 31-run 零等待测试从 `11.103 ms` 降至 `10.710 ms`，E2E 提升 `3.66%`，
  preferred/natural assignment 为 `61/62=98.39%`；
- 独立 11-run sweep 中，`0/200/500 us` 分别为
  `10.687/10.713/10.748 ms`，有限等待没有额外收益。

这证明“尾 expert 在 W13 完成后迁移并扩容 W2”在 planner 已知空闲目标时可实现
稳定局部收益；它不推翻任意 task 自动 regroup 的负结果。该路径仍是实验 bridge，
不进入 production cost-model 搜索。

### 9.18 单次 bounded tail repartition 验证

2026-07-30 在 AmazonC5192Cores NUMA0 CPU `0-95` 上先做静态可行性验证。固定
TP4 H4096/F512、8 个 active expert、每 expert `M=1536`、`R13=2,R2=1` SVE JIT
kernel；首波保持 `6x16T`，两个第二波 expert 同时改为 `24/32/48T`。每轮以两
个 tail 中较慢者决定 wall time，结果为：

| tail 方案 | E2E median | aggregate GFLOP/s | 相对 strict |
| --- | ---: | ---: | ---: |
| strict `2x16T` | 11.143 ms | baseline | - |
| `2x24T` | 9.809 ms | 15.76 TFLOP/s | +13.60% |
| `2x32T` | 10.057 ms | - | +10.78% |
| `2x48T` | 9.902 ms | - | +12.54% |

native stage trace 中，strict 两个 tail 的 W13 为 `3.32/3.38 ms`、W2 为
`1.37/1.35 ms`；24T 降为 W13 `2.358/2.400 ms`、W2 `0.946/0.930 ms`。
48T 的 W2 更快但 W13 回升，因此 24T 在该 workload 最优。完整原始结果见
`optimizations/fused_moe_sve/results/amazon_192c_static_tail_repartition.md`。

production 第一版据此只加入有限候选和结构判定，不硬编码 24T。旧 serialized
isolated formula 对同一 case 预测 strict/24/32/48T 为
`10.996/10.016/9.585/9.633 ms`，能判断 repartition 有收益，但把 32T 排在
24T 前。2026-07-30 的定向采样进一步证明这不只是缺少 24T isolated 点：
`M=1536` isolated 的 24/32/48T 分别为 `3.424/3.261/3.746 ms`，普通双任务
contention 为 `4.508/3.918/3.784 ms`，仍不产生真实 tail 的 24T 排序。原因是
普通 `[t,t]` profile 将 lane 连续放置并只保存宽度 signature，而 bounded tail
固定放在两个 48-core half 的起点；物理 interval 信息在旧模型中丢失。

结构回归覆盖候选开关、恰好一个 terminal wave、blocker overlap、Plan V2 fixed
bridge 和 cache rebuild；Python/native cold planner 在 96-core profile 上逐字段
等价。

随后在同一 NUMA0 上用 production planner 做 51-run 交错 E2E 复测，同时保留
原 strict 和三个显式宽度。auto 选择 `2x32T`，cold/warm planning 分别为
`0.617/0.144 ms`（均在执行计时区间外），执行结果为：

| production 方案 | E2E median | p10--p90 | 相对 strict |
| --- | ---: | ---: | ---: |
| strict `2x16T` | 11.617 ms | 11.457--11.854 ms | - |
| auto `2x32T` | 10.352 ms | 10.118--11.095 ms | +12.22% |
| 显式 `2x24T` | 10.257 ms | 10.148--10.461 ms | +13.26% |
| 显式 `2x32T` | 10.441 ms | 10.217--15.373 ms | +11.26% |
| 显式 `2x48T` | 10.279 ms | 10.182--10.467 ms | +13.01% |

因此 production 候选确实消除了原 fixed-lane 尾部，auto 离本轮最优 24T 仅
`0.93%`。auto 与同构显式 32T 的 median 差 `0.86%`，且显式 32T 存在系统
长尾样本，说明宽度间约 1% 的差异已接近当前 whole-call 噪声；24T
width-local calibration 仍是排序精化项，不影响 bounded repartition 是否有收益
的主结论。

随后使用当前 extension hash
`54a8824324af7e2c017ce4c6d00feade3bc4854a63f2e0db34413dd765ff5b00`
对三个对齐 layout 各采样 101 次：

| tail 方案 | median | p10--p90 | 相对 strict |
| --- | ---: | ---: | ---: |
| strict | 11.137 ms | 11.011--11.469 ms | - |
| `2x24T` | **9.797 ms** | 9.684--9.910 ms | **+13.68%** |
| `2x32T` | 9.997 ms | 9.769--10.667 ms | +11.39% |
| `2x48T` | 9.866 ms | 9.769--9.994 ms | +12.88% |

profile 因而增加 exact-layout bounded-tail anchor。它只命中
`M=1536, root=6x16T, tail_starts=(0,48)` 的三个 measured width；校准后
Python/native planner 均选择 24T。其他 route、异构 route、不同 root shape 或
stage-window override 保持原 simulator 路径。

校准后的 production auto 在同一 NUMA0、当前 extension 上复测 51 次，确认
实际选择 `2x24T`：median `9.842 ms`、p10--p90
`9.719--9.992 ms`，相对 strict `11.103 ms` 提升 `12.81%`。anchor 预测
`9.797 ms`，相对实测 median 的绝对误差为 `0.045 ms`、相对误差为
`0.46%`。未采样的 `M=1548` 不命中 anchor，并回退到 stage-aware simulator。

同一 workload 的下一步实验把两个 terminal expert 各沿 M 切成两个
`M=768` slice。每个 slice 使用 24T，四个 strict task 按 grouped 顺序占用
`[0,24)`、`[24,48)`、`[48,72)`、`[72,96)`，因此 tail wave 不再留下空闲核心。
51-run 结果为：

| tail 方案 | E2E median | p10--p90 | aggregate GFLOP/s |
| --- | ---: | ---: | ---: |
| strict `2x16T` whole expert | 11.122 ms | 10.948--11.382 ms | 13.902 TFLOP/s |
| `2x24T` whole expert | 9.804 ms | - | 15.77 TFLOP/s |
| grouped `4x24T`, `M=768` | **8.710 ms** | 8.647--8.830 ms | **17.751 TFLOP/s** |
| interleaved `4x24T`, `M=768` | 8.755 ms | - | 17.66 TFLOP/s |

grouped M split 相对 whole-expert `2x24T` 的 throughput 提升 `12.56%`，
相对 strict 提升 `27.68%`；输出在正式计时前通过 bitwise equality 检查。
grouped 比 interleaved 快 `0.51%`，因此 exact anchor 同时绑定 route-slice 数
和物理 task 顺序。当前 profile 只增加
`root=6x16T, M=1536, width=24T, route_slices=2` 的 exact anchor；未命中的
route、layout、kernel identity 或 stage-window policy 不生成 route-sliced
候选，而不是把该结果外推到其他 terminal tail。

接入 Python/native production planner 后再次交错运行 51 次，auto 实际返回
`tail_repartition_width=24`、`tail_repartition_route_slices=2` 和 10 个 strict
task。production auto 为 `8.753 ms`（p10--p90
`8.684--8.919 ms`，`17.665 TFLOP/s`），显式 grouped 对照为
`8.746 ms`，两者只差 `0.08%`；相对同轮 strict `11.180 ms` 的 throughput
提升 `27.73%`。anchor `8.710 ms` 对 production median 的误差为 `0.49%`。

### 9.19 Strict 尾部任务领取验证

在 AmazonC5192Cores 的 NUMA0 `0--95` 上，以 TP4
`H=4096, F=512, E=256`、2048 tokens、TopK=6、SVE JIT exact-M 和
`R13=2,R2=1` 路径验证 2.3.2。每组复用同一份 packed weights，strict 与
tail-steal 交错执行；使用 8--10 次 warmup 和每种模式 101 次正式采样，uniform
额外使用 201 次采样。实验参数为 $d=2,r_{\min}=2$。

| workload | Plan V2 shape | strict median | tail-steal median | throughput 收益 | 单次 trace 迁移 |
| --- | --- | ---: | ---: | ---: | ---: |
| `moe256-uniform` | `6x16T` | 16.451 ms | 16.376 ms | +0.46% | 1 |
| `moe256-active-set-128` | `6x16T` | 12.140 ms | 11.897 ms | +2.04% | 0 |
| `moe256-tiered-hotspot` | `12x8T` | 8.861 ms | 8.862 ms | -0.02% | 0 |
| `dsv4-real-2048-seq70` | `12x8T` | 14.791 ms | 14.757 ms | +0.23% | 6 |

`active-set-128` 的单次 trace 没有发生迁移，因此其收益不能归因于 work
stealing；实验分支同时把 fixed executor 的全 task 扫描收窄为 per-team queue，
该 dispatch 差异也包含在 E2E 数字中。captured route 则明确观察到 6 个未启动
whole experts 改在其他 8T team 上执行，说明 suffix migration 在真实不均匀
路由上可达。tiered hotspot 的 planner lane 已平衡，没有可领取后缀，性能保持
中性。

ready-token auto policy 为不均匀 route 启用 drain 时，释放的 team 继续消费原
merge queue；测试同时覆盖 `early_merge=False/True`，输出均与原 strict path
bit-exact。完整 ARM MoE 测试为 `104 passed, 1 skipped`。

初版每次调用重新读取 96 个 CPU 的 sysfs NUMA topology，使
`plan_validate` 从约 `0.026 ms` 增至 `0.240 ms`，掩盖了迁移收益。缓存
CPU-to-NUMA 映射后，代表 trace 的验证阶段为 `0.017 ms`。`d=4` 在 captured
route 中增加了迁移数，但未增加中位收益；$r_{\min}=1$ 同样只增加领取次数，
未改善 uniform E2E，因此实验默认采用更保守的 $d=2,r_{\min}=2$。

当前数据只证明该机制在既有上层计划后的 residual tail 上无显著回退并可获得
小幅收益；它还不是 cost-model 可见的 production 决策。功能保持默认关闭，
后续需先把相同 suffix state machine 加入事件模拟，再决定是否由 planner
显式启用。

### 9.20 Cold-phase CP-SAT mixed-width oracle

2026-07-31 使用 AmazonC5192Cores 双 NUMA TP4/F512 exact-M `R13=2,R2=1` profile，
对单 rank 的 96 cores 离线求解 6.2.1。默认 stage-window policy 为
`amazon_c5_192c_tp4_f512_v1`，width 域为 `1,2,4,8,16`，fixed baseline 为
当前 DSV4 strict `12x8T` lane DAG。packed-B 带宽 ceiling 取该机单 NUMA
STREAM Triad 的 `336.4 GB/s`，时间/bandwidth quantum 分别为
`1000 ns/0.25 GB/s`；没有设置 cold stream 数硬上限。该实验不执行 kernel，
只消费已提交 profile 和 workload histogram。

`expert` 流体聚合结果如下。区间为 CP-SAT 的 best bound 到 incumbent；收益区间
按 6.2.1 的交叉上下界计算。DSV4 合并同一模型 15 s 与 60 s 两轮中更强的 bound
和 incumbent，其余 case 为 15 s、8 solver workers。

| workload | fixed `12x8T` LB--UB | mixed LB--UB | mixed 相对 fixed 收益区间 | incumbent 对比 |
| --- | ---: | ---: | ---: | ---: |
| `dsv4-real-2048-seq70` | `11.214--12.500 ms` | `8.345--8.782 ms` | `27.7%--49.8%` | `42.3%` |
| `moe256-long-short-bimodal` | `11.368 ms` exact | `6.699--6.992 ms` | `62.6%--69.7%` | `62.6%` |
| `moe256-tiered-hotspot` | `6.510 ms` exact | `5.409--6.266 ms` | `3.9%--20.4%` | `3.9%` |

DSV4 最好 mixed incumbent 的 width histogram 为
`173x1T + 7x2T + 7x4T + 34x8T + 2x16T`。这说明模型利用的主要自由度不是把
所有 expert 统一缩窄，而是让大量短/中 expert 以窄 team 填入长 expert 的
steady 区间，同时只为少数长或尾部 task 使用宽 team。

同一 DSV4 case 的 `range` 粒度在 15 s 内只找到 `13.152 -> 12.971 ms`
（`1.4%`）的 incumbent 改善，但 mixed solver gap 仍为 `35.7%`，收益上界仍为
`57.6%`，因此该点不能反证 fluid headroom；它只说明完整 range-level optional
interval 模型尚未在当前时限收敛。`300/336.4/375.9 GB/s` 的 10--15 s
敏感性 sweep 中，DSV4 mixed incumbent 分别为 `9.974/8.782/8.396 ms`，表明
绝对上限依赖选用的 DRAM ceiling；solver gap 不同，不能把三点的收益百分比拟合成
硬件规律。

结论是：在 cold packed-B 流体 surrogate 内，captured mixed distribution 相对
固定 `12x8T` 至少仍有约 `27.7%` 的已证明调度 headroom；但这不是可直接宣称的
E2E 收益。离线命令、模型边界和结果解释见
`optimizations/fused_moe_sve/results/amazon_192c_cold_phase_cp_sat_oracle_20260731.md`。

#### 9.20.1 Mixed incumbent 的 runtime 反证

同日将 DSV4 mixed incumbent 按 6.2.2 降为连续 core strict Plan V2。物理 placement
在 `0.06587 s` 内证明 `OPTIMAL`，223 个 task 的 width histogram 保持
`173x1T + 7x2T + 7x4T + 34x8T + 2x16T`；所有 variant 先通过 bit-exact
输出对比。AmazonC5192Cores NUMA0 `0-95` 上随机交错测量 7 次 warmup、51 次 run：

| runtime plan | median | p10--p90 | aggregate | 相对 fixed throughput |
| --- | ---: | ---: | ---: | ---: |
| fixed `12x8T` | `14.854 ms` | `14.803--14.924 ms` | `10.409 TFLOP/s` | baseline |
| mixed eager | `20.643 ms` | `20.528--21.609 ms` | `7.490 TFLOP/s` | `-28.05%` |
| mixed release `0.25x` | `20.649 ms` | `20.549--21.444 ms` | `7.488 TFLOP/s` | `-28.07%` |
| mixed release `0.50x` | `20.765 ms` | `20.676--20.915 ms` | `7.446 TFLOP/s` | `-28.47%` |
| mixed release `0.75x` | `20.780 ms` | `20.711--20.961 ms` | `7.441 TFLOP/s` | `-28.52%` |
| mixed release `1.00x` | `20.869 ms` | `20.793--20.994 ms` | `7.409 TFLOP/s` | `-28.82%` |
| mixed release `1.25x` | `21.008 ms` | `20.905--21.615 ms` | `7.360 TFLOP/s` | `-29.30%` |

oracle incumbent 预测 fixed/mixed 为 `12.500/8.782 ms`；实测误差分别为
`+18.83%/+137.63%`。mixed 没有得到预测的 `42.34%` throughput headroom，反而
增加 `40.50%` latency。`0.25x` release 相对 eager 只增加 `0.03%`，因此回退不由
release gate 或读时钟开销导致；继续增加 delay 只会单调变慢。

这组反证关闭了首版 cold-only surrogate 作为可执行候选生成器的结论。大量 1T
expert 并发时，首个 M12 phase 之后不能维持 isolated cache 状态；当前 oracle 没有
计入整个 active window 上的 LLC-to-L2 service、容量淘汰、active-set compute/frequency
slowdown、dispatch、scratch/gather、store 和 merge。非零 release 保持实验诊断能力，
但不进入 production planner。下一版 oracle 必须在完整 active lifetime 上建模 lower
cache service，并先通过 production contention model 过滤候选。

2026-08-01 的 trace、width sweep 和 PMU follow-up 将退化进一步定位如下：

1. `scheduled_compute` 从 fixed 的 `14.110 ms` 增至 mixed eager 的 `19.848 ms`，
   已解释总回退中的约 `5.74 ms`；route build、merge 和其他外层阶段合计没有形成
   可见差异。
2. fixed 的实际平均 active cores 为 `93.51/96`，mixed eager 只有 `57.18/96`。
   mixed 虽然可见 W13/W2/gather core-time 从 `1228.7` 降至 `1108.3 core-ms`，但
   固定 core rectangle 无法在 duration 失配后回填空洞，wall time 因低占用反而增加。
3. mixed 的 W13/W2 实际活跃窗口峰值为 `201.5/161.5 MiB`，fixed 仅为
   `44.0/36.0 MiB`；目标 NUMA 的共享 L3 为 `96 MiB`。对应 PMU 每次调用的
   L2 refill 从 `5.208` 增至 `6.203 GiB`（`+19.1%`），last-level read miss 从
   `0.251` 增至 `0.305 GiB`（`+21.5%`）。
4. 受限 width oracle 中，`min-width=2/4/8` 分别实测 `17.114/15.087/15.833 ms`；
   `4T` 最接近 fixed（throughput `-2.00%`），其 W13/W2 峰值窗口降为
   `47.5/33.5 MiB`。这验证了该机上至少 `4T` 同时兼顾约 24 个 active expert、
   96-core 覆盖和 LLC window；`8T` 又损失 tail packing，`2T` 仍有过多 active window。
5. isolated duration 误差随 route/width 显著变化：1T 的 M1/M28/M197 actual/model
   中位数分别为 `3.23x/4.58x/1.39x`；M28 runtime 同时活跃最多 33 个 expert。
   oracle 计划的 M<=12 cold 并发最多 8 个，runtime 却达到 20 个，说明绝对 release
   不能在 task 变慢后维持计划中的并发上界。
6. 只给 M<=32 的 1T task 增加 completion-token slot cap，24 slots 将 mixed 从
   `20.661` 改善到 `19.316 ms`，但 8 slots 退化到 `26.692 ms`。因此容量争用确实是
   因素，但单独串行化会进一步降低 core occupancy；必须同时调整 team width/placement。
7. OS context switches 约 `436--462/call` 且 migration 为 0。按 core 预建 task list
   使 mixed 退休指令减少 `36.8%`，wall time 仍保持约 `20.7 ms`；它只移除了空闲核的
   扫描工作并暴露 `34.75%` active memory-stall，不是 critical-path 根因。

因此当前可执行证据支持的机制链为：cold-only duration/resource 低估 -> 真实窄任务
延长并重叠 -> active packed-B window 超过 L3 且 refill 增加 -> 静态 rectangle 出现
无法回填的 core holes -> 平均占用从约 97% 降至约 60%。下一版候选必须同时满足
完整 stage-window resource 评分和 runtime 可回填性；不能只优化 cold DRAM 时间轴。

完整命令和机器可读样本见
`optimizations/fused_moe_sve/results/amazon_192c_cold_phase_oracle_runtime_20260731.md`
与同名 `.json`。

### 9.21 硬件容量派生的 L1-hot GEMM core 校准

2026-08-02 在 AmazonC5192Cores NUMA0 `0-95` 上，将解析模型的计算上界从
register-only M12 BFMMLA 改为 8.2.4 定义的 L1-hot M12 full-no-store GEMM。
sysfs 实测 L1D/L2/LLC 为 `64 KiB / 2 MiB / 96 MiB`，探针几何自动选择为
`M12/K728/N16`（A+B `40,768 B`）和 `M12/K18720/N16`
（A+B `1,048,320 B`）。32 MiB HugeTLB、显式绑核下关键 service 为：

| threads | register-only BFMMLA | L1-hot GEMM core | L2-hot GEMM diagnostic |
| ---: | ---: | ---: | ---: |
| 1 | `0.412` TFLOP/s | `0.340` TFLOP/s | `0.335` TFLOP/s |
| 8 | `3.289` TFLOP/s | `2.712` TFLOP/s | `2.646` TFLOP/s |
| 24 | `9.865` TFLOP/s | `8.154` TFLOP/s | `7.976` TFLOP/s |
| 48 | `19.539` TFLOP/s | `16.244` TFLOP/s | `15.822` TFLOP/s |
| 96 | `39.309` TFLOP/s | `30.377` TFLOP/s | `28.215` TFLOP/s |

L1-hot 探针因单次只有约 `0.8 us`，使用 64 次 warmup 和 4096 次 timed call，避免
慢线程离群样本压低高核数曲线。因此 `39.309 TFLOP/s` 只能表示寄存器矩阵指令上限；实际 production loop 的
多发射、A/B load、地址更新和分支成本已由 `gemm_core_flops` 吸收。使用相同 12 个
isolated residual training points 重算后，108 个 isolated holdout 的 MAPE 从
`10.22%` 降到 `9.18%`，首次通过 isolated 10% 单项门槛；contention P90 从
`47.79%` 变为 `49.57%`，最大 shape regret 保持 `8.17%`。计算 ceiling 的物理语义
更正确，common stage residual 从 `1.18546` 降到 `1.10842`，但 per-route residual
仍为 `418.76 ns`，说明非 GEMM operator work 尚未拆成独立资源。模型仍未通过
contention/regret production gate；主要剩余误差是多 team packed-B retention 和
NUMA 内 LLC topology，而不是重新提高 matrix peak。

同日用整批计时复核 V3 高核数曲线：每个 worker 在一次 native call 内连续执行
131,072 个不逐次计时的 L1-hot kernel，只在 batch 外计时，随机顺序重复 5 次。
L1-hot 在 `1/48/64/80/96T` 分别为
`0.345/16.524/21.168/25.820/30.438 TFLOP/s`，相对单核线性效率为
`100.00/99.69/95.78/93.46/91.82%`；96T 五次范围仅
`30.406--30.447 TFLOP/s`。相同 batch 口径的 register-only BFMMLA 在线程宽度
`1/48/64/80/96T` 的线性效率为
`100.00/99.53/99.36/99.21/98.95%`。因此 V3 的 matrix execution units 本身
没有显著跨核共享上限，但完整 L1-hot load/control/compute loop 在 48--64 核之间
出现可复现拐点；它不能被逐调用时钟或单次调度离群解释。sysfs 只暴露 private
L1/L2 和覆盖 `0-95` 的单一 L3 domain。进一步用 V3 implementation-defined PMU
事件复核后，可以排除 cache capacity 和普通 DVFS：48/96T 的
`L1D_CACHE_REFILL/L1D_CACHE` 分别只有 `0.00545%/0.00496%`，
`STALL_BACKEND_L2D=0`，`CPU_CYCLES/CNT_CYCLES` 均为 `3.300`，且每 FLOP
retired instructions 不变。相反，每 FLOP backend stall slots 增加 `29.2%`，
`STALL_BACKEND_CPUBOUND` 增加 `41.4%`，`STALL_BACKEND_BUSY` 增加 `87.0%`；
其子事件主要是 vector issue queue full，`DISPATCH_STALL_IQ_VX/FLOP` 增加
`41.2%`，而 LS issue queue 绝对计数小约 160 倍。register-only 对照的
`IQ_VX/FLOP` 在 48--96T 变化小于 `0.1%`。

48T victim/aggressor 实验给出同一结论：48 个 full-loop victim 单独运行的 worker
中位时间为 `1.6794 s`；另加 48 个 register-only worker 后为 `1.6864 s`
（`+0.42%`），另加 48 个 full load/BFMMLA worker 后为 `1.8308 s`
（`+9.01%`）。因此可观测机制是 socket-wide、由 load-to-vector-compute 混合流
触发的 vector-dispatch backpressure，不是私有 cache miss、L3 data fabric 或
BFMMLA 单元共享上限。平台没有暴露可以把 firmware dispatch/power throttling 与
其他实现特定 vector-issue control 进一步区分的计数器，所以最后一层硬件命名仍是
推断。模型不应把 V3 视为全范围严格线性，也不应使用单一 power curve 将
64--96T derate 扩散到 1--48T；后续应改为薄校准的分段 active-core efficiency。

同日的 M12 双 B 寄存器 column-pipeline 反证了仅靠局部指令重排即可关闭该
derate 的假设。21 轮长窗口中，96T 从 `30.420` 提升到 `30.617 TFLOP/s`
（`+0.648%`），线性效率从 `91.681%` 提升到 `92.273%`；但
`DISPATCH_STALL_IQ_VX` 反而增加约 `37.5%`。因此保留现有分段 active-core
efficiency 建模：这个重排只减少约 `0.74%` 的 active-core cycles，不足以改变
service curve 或 production calibration。

详细命令、完整 service 表和 artifact 路径见
`optimizations/fused_moe_sve/results/amazon_192c_analytic_hot_gemm_core_20260802.md`。

### 9.22 Neoverse-V1 的 L1/L2 容量派生复验

同日用相同探针在 AmazonECS8Cores 的 8 个 Neoverse-V1 核上复验。sysfs 实测
L1D/L2/LLC 为 `64 KiB / 1 MiB / 32 MiB`，因此 L1 几何仍为
`M12/K728/N16`（A+B `40,768 B`），L2 几何自动缩为
`M12/K9360/N16`（A+B `524,160 B`）。长采样结果为：

| threads | register-only BFMMLA | L1-hot GEMM core | L2-hot GEMM diagnostic |
| ---: | ---: | ---: | ---: |
| 1 | `0.331` TFLOP/s | `0.289` TFLOP/s | `0.302` TFLOP/s |
| 2 | `0.662` TFLOP/s | `0.483` TFLOP/s | `0.495` TFLOP/s |
| 4 | `1.058` TFLOP/s | `0.872` TFLOP/s | `0.890` TFLOP/s |
| 8 | `1.833` TFLOP/s | `1.621` TFLOP/s | `1.663` TFLOP/s |

register-only 在 1T/8T 分别高估完整主循环 `14.4%/13.1%`，支持继续将
L1-hot full-no-store 作为可执行 compute resource。L2-hot 比 L1-hot 高
`2.1%--4.5%`，原因是 `K=9360` 比 `K=728` 更充分摊薄固定调用和循环边界成本；
因此当前 L1-hot 是保守的工程 ceiling，不是严格数学 peak。若后续需要更紧的上界，
应在一次 JIT 调用内重复同一 L1-resident K chunk，而不是恢复 register-only 值。
V1 的 8T L1-hot 仅为 1T 的 `5.60x`，register-only 也仅为 `5.54x`；这证明多核
compute derate 在无 lower-cache 流量时已经存在，必须保留在线程宽度 service curve
中，不能归因给 DRAM contention。

完整命令、重复性说明和原始 artifact 见
`optimizations/fused_moe_sve/results/amazon_v1_8c_analytic_hot_gemm_core_20260802.md`。

### 9.23 纯 GEMM 四状态分解验证

2026-08-02 在 AmazonC5192Cores NUMA0 `0-3` 上增加 full-no-store 纯 GEMM
验证。该实验不改变 8.2 的 active phase 公式，而是检查更细的 M-panel/N-tile
状态分解能否作为后续 service-cost 细化。对每线程一个 stage window，令
$P=M/12$，$Q$ 为该线程负责的 N8 tile 数，则当前 M-panel 外层、N-tile 内层
循环的精确访问状态计数为：

$$
C_{cc}=1,\qquad C_{hc}=Q-1,\qquad C_{ch}=P-1,\qquad
C_{hh}=(P-1)(Q-1),
$$

其中首字母表示 A、次字母表示 B，$c/h$ 分别表示本次 tile 访问前 cold/hot。
多个 W13 stage window 只将四个计数同时乘以 window 数。该计数来自循环偏序，
不需要机器校准；每个状态的 service time 才需要薄校准。

4T 的 M12/K728/N16 两 tile probe 重采得到 A-hot/B-stream、
A-stream/B-hot、A+B-stream 分别为 `1.055/0.778/0.829 TFLOP/s`。结合 L1/L2-hot
GEMM service 后，K728 每个 N8 tile worker-wave 的
`cc/hc/ch/hh(L2)` 成本为 `0.819/0.530/1.026/0.422 us`。将成本按 K 线性缩放，
再对 TP4 的 W13 `K4096,N128/thread,2 windows` 和 W2
`K512,N1024/thread,1 window` 做纯 GEMM 验证，得到：

| M | predicted | measured | error |
| ---: | ---: | ---: | ---: |
| 12 | `0.147 ms` | `0.160 ms` | `-8.48%` |
| 24 | `0.268 ms` | `0.321 ms` | `-16.53%` |
| 48 | `0.510 ms` | `0.556 ms` | `-8.15%` |
| 192 | `1.966 ms` | `1.942 ms` | `+1.23%` |
| 768 | `7.786 ms` | `7.502 ms` | `+3.79%` |
| 2040 | `20.641 ms` | `19.685 ms` | `+4.85%` |

每点为 11 个同步 4-process wave 的中位数；每个 worker/wave 使用从未计时访问过的
独立 packed-B expert，32 MiB HugeTLB，显式绑核和 NUMA-local 内存。全表绝对误差
中位数为 `6.50%`、最大 `16.53%`；限制到 M>=192 后为 `3.79%/4.85%`。

因此四状态**计数**可保留为解析骨架，但当前单组 service cost 不能跨工作集几何直接
线性迁移：K728 的 A+B footprint 可放入 L1，而 W13 K4096 的单 M12/N8 tile 不可；
W2 K512 的固定 control/address/call 成本不随 K 同比缩小；M24/N>=128 的已知低谷
也不能由独立 M12 状态相加表示。长 route 的误差方向反转，是因为 no-reuse A-stream
校准比刚完成 pack、随后反复扫描的 packed A 更悲观。该验证当前只作为 shadow
diagnostic；不修改 production empirical backend、analytic phase service、候选空间或
剪枝。后续若接入，必须先按 A/B 实际 cache level 和 K-independent fixed cost 分层，
并重新通过 isolated/contention/regret gate。

完整表、命令和误差归因见
`optimizations/fused_moe_sve/results/amazon_192c_gemm_memory_services_20260802.md`。

### 9.24 短 route packed-B 窗口与每线程窗口不变量

2026-08-06 在 AmazonC5192Cores NUMA0 `0-95` 上标定 8.2.5 的
$p_{\mathrm{eff}}(\omega)$，并对 production planner 做端到端 A/B。形状为 TP4
`H=4096/F=512`，每 expert 12 MiB packed BF16，单个 stage window 4 MiB，
`R13=2,R2=1`、SVE JIT exact-M、32 MiB HugeTLB、NUMA-local。每个计时 task 使用不同
expert，因此 packed-B 始终为流式读取。有用带宽只计一次 compulsory 权重字节；
本节点实测峰值为 `367.4 GB/s`。

隔离扫描使用 192 个 expert，因为 192 能被全部实测 lane 数（96/48/24/12）整除。
早前的 195-expert 扫描保留在数据目录中，但不可用于宽度比较：多出的 3 个 task
使临界路径随 lane 数变化，最多给宽 team 带来 4 个百分点的偏置。

$M=28$（panel 为 `12+12+4`）按相同每线程窗口 $\omega$ 对齐四种宽度，有用 GB/s：

| $\omega$ | `96x1T` | `48x2T` | `24x4T` | `12x8T` |
| ---: | ---: | ---: | ---: | ---: |
| 4 MiB | 123.8 | | | |
| 2 MiB | 133.0 | 128.4 | | |
| 1 MiB | 203.7 | 203.6 | 209.7 | |
| 0.5 MiB | 236.1 | 241.7 | 249.4 | 240.5 |
| 0.25 MiB | 285.0 | 297.7 | 300.8 | 290.8 |
| 0.125 MiB | | 298.9 | 300.8 | 291.7 |
| 0.0625 MiB | | | 297.7 | 287.7 |
| 0.03125 MiB | | | | 280.5 |

同一 $\omega$ 下四种宽度差异为 `3.0%--5.6%`，而 $\omega$ 本身跨越
`123.8--300.8 GB/s`（`2.43x`）。因此 $\omega$ 是主导变量、宽度是二阶效应。对应
的 $p_{\mathrm{eff}}$ 为 `2.97/2.76/1.75/1.47/1.22/1.22/1.23`：上界
$\lceil28/12\rceil=3$ 在 $\omega=4$ MiB 处取到，平台值约 `1.22` 从
$\omega=0.25$ MiB 开始，残余部分归因于 range dispatch、A 重扫和第三个 `M4`
panel，不属于 packed-B 重读。

$M=12$ 对照（$P=1$，无 packed-B 复用可保护）：`96x1T` 在
`legacy/2/1/0.5/0.25 MiB` 下为 `364.7/367.4/366.2/365.4/365.3 GB/s`，极差
`0.7%`；`24x4T` 仅在 $\omega=0.0625$ MiB 处回退 `4.0%`。因此上表的效应确实来自
panel 复用，而非通用的窗口尺寸伪影。

全核忙时聚合活跃窗口恒为 $C\omega$，因此只扫 $g_s$ 无法区分私有 L2 与共享 LLC。
固定 $t=4T$、按 `3/6/12/24` lane 改变活跃核数 $C$（192 experts 对全部 lane 数
整除），并用同点的 $M=12$ 归一化掉该核数下的 DRAM/MLP 上限，得到
$p_{\mathrm{eff}}=B_{M12}/B_{M28}$：

| $\omega$ | `C=12` | `C=24` | `C=48` | `C=96` |
| ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 2.35 (agg 12 MiB) | 2.11 (24) | 1.81 (48) | 1.73 (96) |
| 0.5 MiB | 2.64 (6) | 2.19 (12) | 1.68 (24) | 1.46 (48) |
| 0.25 MiB | 2.14 (3) | 1.76 (6) | 1.25 (12) | 1.21 (24) |
| 0.125 MiB | 2.12 (1.5) | 1.79 (3) | 1.20 (6) | 1.20 (12) |

共享 LLC 容量假设要求 $p_{\mathrm{eff}}$ 只依赖聚合窗口。实测在聚合 12 MiB 的四个
组合上为 `2.35/2.19/1.25/1.20`（相差 2 倍）且严格跟随 $\omega$；反之固定 $\omega$
时聚合变化 8 倍只使 $p_{\mathrm{eff}}$ 变化 `1.36--1.77` 倍，**且聚合压力越大越好**。
因此该假设以相反符号被否证，$\omega$ 仍是主导变量。两项效应可分离且相乘：
$C=96$ 下 $\omega$ 由 1 降到 0.125 MiB 给出因子 `1.44`，$\omega=1$ MiB 下 $C$ 由 12
升到 96 给出因子 `1.36`，预测角点 $2.35/(1.36\times1.44)=1.20$ 与实测
`(C=96, ω=0.125)` 完全一致。由此得到两个结论：$g(M,t)$ 无需活跃核数项，且 96 核
标定在更少活跃核时只会偏保守；实测有效驻留容量约 0.25 MiB/线程，即标称 2 MiB
私有 L2 的约 1/8。残余 $g(C)$ 的机制未定，候选为高内存压力下预取被节流，或
$M=28$ 的三段 panel 循环固定开销在低 $C$（每 expert 0.437 ms vs 1.436 ms）占比更高
造成的比值抬升；区分需要 PMU 或独立的每 expert 固定成本标定。

端到端 A/B 使用 production planner、5 warmup 与 31 次交错计时，全部变体在计时前
与关闭 stage-window policy 的 `legacy` 计划 bit-exact。候选 band 为
$13\le M\le48$，`1T/2T/4T/8T` 分别取 `0.25/0.5/1/1 MiB`，W13 与 W2 同值，
$M\le12$ 不覆盖：

| workload | band 内 expert / route | `policy_v1` | 候选 band | 提升 |
| --- | --- | ---: | ---: | ---: |
| `dsv4-real-2048-seq70` | 120 / 3314 (27%) | 15.155 ms | 14.116 ms | `+7.36%` |
| `moe256-uniform` | 256 / 12288 (100%) | 17.000 ms | 13.688 ms | `+24.20%` |
| `moe256-long-short-bimodal` | 0 / 0 | 11.405 ms | 11.415 ms | `-0.09%` |
| `moe256-active-set-128` | 0 / 0 | 9.646 ms | 9.630 ms | `+0.17%` |
| `moe256-tiered-hotspot` | 0 / 0 | 8.379 ms | 8.351 ms | `+0.34%` |

反序交错复测给出 `dsv4 +7.83%` 与 `uniform +25.17%`，与正序差不超过 0.9 个
百分点。把窗口直接覆盖到 `legacy` 计划上（task 图、宽度、placement 逐字节相同）
与让 cost model 重新评分在每个 workload 上相差不超过 `0.2%`，且所选 shape、
execution mode 与 `policy_v1` 完全一致；因此收益全部来自窗口值，不来自候选重搜。

两点边界必须记录。第一，现有 policy 的首个 band 为 `min_routes=49`，而
`moe256-uniform` 恰好是 $M=48$，因此 `policy_v1` 覆盖 0 个 task、只有 `0.06%`
收益；这是覆盖缺口而非标定上限。第二，`results/amazon_192c_weight_windows.md`
给出的"每线程约 1 MiB"最优是在 $M=2040$ 上标定的，那里 A 为每 range 16.7 MB，
range 增殖代价高；该不变量不能外推到短 route，本节实测的短 route 最优为
$\omega=0.125$--$0.25$ MiB。

该 band 已在 v0.67 落地为 production 默认 `amazon_c5_192c_tp4_f512_v2`，写作
band 级 $\omega=1/4$ MiB、`widths=(1,2,4,8)`、$t=8$ 取 $1/8$ MiB。落地后在同一
session 重测的默认对默认提速为 `dsv4-real-2048-seq70` $7.59\%$、
`moe256-uniform` $24.29\%$；三个无 band 内 expert 的 workload 变化在
$\pm0.33\%$，而同期 `legacy` 变体（不使用 policy，因此可证明不受影响）自身摆动
$-0.72\%$--$+0.38\%$，即噪声底高于这三个 workload 的位移。`--reverse-order`
交叉验证给出 $7.38\%$ 与 $23.67\%$，同量级。

仍未完成的标定有三项：W13/W2 分离的二维标定（当前 band 让两个 stage 共用一个
$\omega$，故收益是下界）、`16T` 及更宽宽度的覆盖（扩覆盖集会翻转
`can_use_full_workload_anchor`，必须实测而不能由 $\omega$ 不变性推断），以及至少
一台其他机器/并行度的重复测量。
完整表、命令、逐次样本和数据文件见
`optimizations/fused_moe_sve/results/amazon_192c_short_route_stage_windows_20260806.md`。

#### 9.25 W13/W2 每线程窗口的二维标定

8.2.5 预测两个 stage 的 $\omega^\ast$ 因 shared-A 相差 $H/F$ 倍而不等。为分离这
两个轴，`profile_heterogeneous_overlap.py` 增加 `--small-w2-window-sweep`，把
$\omega_{W13}$ 与 $\omega_{W2}$ 做叉乘。per-task stage window 只存在于 Plan V2，
因此该模式改走 `fused_moe_bf16_tiled_async_plan`；operator-wide 预算路径不变。
测量在 AmazonC5192Cores NUMA0 上固定 `24x4T`（96 核全忙）、192 个同构 expert
（整除 lane 数），扫 $M\in\{13,28,48,120,320\}$。

结果表见 8.2.5。三项结论：

1. **三个 production band 值被独立复现**。$M=28$（新 `13--48` band）、$M=120$
   （`96--143`）、$M=320$（`288--575`）的二维最优精确等于已标定的
   $(\omega_{W13},\omega_{W2})$，而这些值当初是按每 range 字节逐格搜索得到的。这
   同时验证了标定表和 $\omega$ 参数化。
2. **$\omega_{W2}$ 是弱轴，8.2.5 的"收益是下界"这一保留可以撤销**。固定最优
   $\omega_{W13}$ 后，$\omega_{W2}$ 在 3--4 个档位上的极差多在 $1.5\%$ 以内；
   较大的极差全部来自 $\omega_{W2}=1/2$ MiB 处的悬崖，而非最优点附近的坡度。因此
   两个 stage 共用一个 $\omega$ 并没有实质性低估收益。
3. **`13--48` band 的 $\omega_{W13}$ 不改**。$M=13$ 与 $M=48$ 的孤立最优是
   $1/8$ MiB（比现值好 $1.36\%$/$2.95\%$），但 $M=28$ 的最优是现值 $1/4$ MiB
   （$1/8$ 差 $0.5\%$），band 只能取一个。端到端验证 $\omega_{W13}=1/8$：
   `dsv4-real-2048-seq70` 对 legacy 为 $7.38\%$ 对现值 $7.25\%$，在噪声内；
   `moe256-uniform` 从 $13.693$ 改善到 $13.513$ ms（$+1.33\%$），但**唯一原因是
   cost model 把 shape 从 `8T` 翻到 `1T`**——同一次运行里保持 legacy shape 的
   `manual` 变体反而从 $13.686$ 退化到 $13.803$ ms。收益低于 2% 采用门槛，且该
   `1T` 形状在此窗口下没有孤立标定支撑，故不落地。

数据在 `optimizations/fused_moe_sve/results/data/stage_window_omega_20260807/w2_2d_m{13,28,48,120,320}.json`
与 `w13_eighth_{preset}.json`。

#### 9.26 宽度不变性的边界与 legacy 几何的自动缩放

8.2.5 的 $\omega$ 不变性是在 `1T`--`8T` 上测的。固定 $\omega$ 扫六个宽度显示它不能
外推：$M=28$ 在 $\omega=0.25$ MiB 下由 `4T` 峰值到 `16T`/`32T` 分别掉
$13.0\%$/$27.0\%$，$M=120$ 在 $\omega=0.125$ MiB 下掉 $19.7\%$/$39.0\%$，而
`1T`--`8T` 的极差只有 $4.2\%$/$11.4\%$。超过 8 线程后宽度本身的代价（队内同步、
并发 expert 数塌到 6 或 3）超过窗口能挽回的量，因此 $\omega$ 在宽 team 上不是
可迁移量。

这一点不影响 policy，因为**operator-wide baseline 几何本身就是一个随宽度缩小的每
线程窗口**：`R13=2,R2=1` 给出一个 $4$ MiB 的 W13 range 与一个 $4$ MiB 的 W2 range，
故 $\omega_{\text{baseline}}=4\ \mathrm{MiB}/t$，即 `1T` 为 $4$ MiB 而 `32T` 为
$0.125$ MiB。宽 team 从来不在病态区：在 `16T`/`32T` 上最优窗口相对 baseline 只值
$+2.4\%$/$+0.8\%$（$M=28$）与 $+4.2\%$/$+8.5\%$（$M=120$，此时 baseline 的
$0.125$ MiB 反而略偏小）。因此 `16T` 及更宽宽度保持继承。

真正的空洞在 `49--95` band 的窄端——V1 只标定了 `8T`。$M=72$ 上 baseline 对最优
窗口为 `1T` $3.39\times$、`2T` $2.94\times$、`4T` $1.65\times$，因为那里
$\omega_{\text{baseline}}$ 分别是 $4$/$2$/$1$ MiB，比最优大 $32$--$16$ 倍。
`amazon_c5_192c_tp4_f512_v3` 用实测最优（`1T`/`2T` 取 $1/8$ MiB，`4T` 取
$1/16$ MiB）补齐这三格，`8T` 逐字节保持 V1 的标定值。catalog 里没有 preset 会把
$49\le M\le95$ 的 expert 排到 8 线程以下，因此 V2 与 V3 在五个 preset 上生成
**逐元素相同的 plan**，墙钟差异（最大 $1.26\%$）按定义是噪声。收益是潜在的：这三
格现在有标定窗口而不是大 $32$ 倍的继承值，且 cost model 不再用 full-workload
anchor 给它们评分。

数据在 `optimizations/fused_moe_sve/results/data/stage_window_omega_20260807/omega_inv_m{28,120}_t{1,2,4,8,16,32}.json`、
`wide_m{28,120}_t{16,32}.json`、`band4995_m72_t{1,2,4,8}.json` 与 `v3_{preset}.json`。

#### 9.28 $\omega^\ast(M)$ 的门限定位与 `144--287` band 的拆分

8.2.5 的两项竞争（B 重读想要小 $\omega$、A 重扫想要大 $\omega$ 且代价 $\propto M/\omega$）
预测存在一个门限：每线程在一个 range 内要扫完整 shared-A，大小 $2MK_s$；它装得进私
有 L2 时跨 range 不重取、A 项近似消失，装不进时代价正比于 $R$。对 W13（$K=H=4096$）
门限落在 $M=2\cdot2^{20}/(2H)=256$，对 W2（$K=F=512$）落在 $M=2048$，两者之比正是
$H/F=8$。

「每线程读全部 A 行」成立的前提是 team 沿 **N 轴**分割（`kN`）。本节全部测点为
$M\le320$、$t\in\{1,2,4,8\}$，按 9.29 的分割规则除 $M\le32$ 且 $t=2$ 外均为 `kN`，
故门限推导在本节适用域内成立。`kM` 区间的 $\omega$ 语义见 9.29。

在 `24x4T`、192 同构 expert、$\omega_{W2}$ 固定 $1/8$ MiB 下扫 $\omega_{W13}$，
有用 packed-B 带宽（GB/s）：

| $M$ | $A_{W13}/L2$ | 1/16 | 1/8 | 1/4 | 1/2 | 1 | $\omega^\ast_{W13}$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 0.47 | 169.8 | **174.2** | 167.1 | 143.7 | 132.0 | 1/8 |
| 160 | 0.62 | 124.9 | **128.7** | 127.6 | 112.1 | 102.9 | 1/8 |
| 176 | 0.69 | 114.0 | **117.8** | 116.4 | 106.5 | 95.9 | 1/8 |
| 192 | 0.75 | 105.5 | **108.2** | 107.9 | 102.7 | 91.3 | 1/8 |
| 200 | 0.78 | 99.8 | 103.1 | **103.3** | 99.2 | 87.6 | 1/4 |
| 224 | 0.88 | 86.7 | 88.2 | 88.9 | **89.6** | 81.4 | 1/2 |
| 240 | 0.94 | 78.5 | 76.0 | 79.4 | **84.1** | 78.2 | 1/2 |
| 256 | **1.00** | 68.2 | 63.5 | 71.5 | **78.3** | 74.6 | 1/2 |
| 300 | 1.17 | 49.4 | 47.7 | 61.0 | **67.5** | 66.1 | 1/2 |
| 320 | 1.25 | 44.6 | 44.0 | 56.8 | **63.0** | 61.8 | 1/2 |

门限得到证实：$\omega^\ast$ 随 $A_{W13}/L2$ 单调爬升 $1/8\to1/4\to1/2$ 并在
$A_{W13}/L2=1.00$ 处到顶。不是突变而是两级斜坡，起点在 $A/L2\approx0.62$--$0.78$，
与 A 并不独占 L2（packed B、intermediate 与 C 共享）一致。$1/4$ 的"平台"实际只是
$M=200$ 一个点、且只比 $1/8$ 好 $0.2\%$，因此不值得单独设 band。

门限与 team 宽度无关，这正是机制的预测——每线程都扫完整 A，故 $2MK_s$ 不含 $t$。
$M=256$ 上四种宽度全部峰值在 $\omega_{W13}=1/2$ MiB：

| $t$ | 1/8 | 1/4 | 1/2 | 1 | 旧 band 值 $1/8$ 的代价 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.6 | 55.2 | **67.1** | 65.0 | $-61\%$ |
| 2 | 62.8 | 68.2 | **72.8** | 71.1 | $-16\%$ |
| 4 | 63.5 | 71.5 | **78.3** | 74.6 | $-23\%$ |
| 8 | 63.1 | 73.5 | **80.1** | 74.1 | $-27\%$ |

由此暴露 V1 表的一个缺陷：band `144--287` 整段规定 $\omega_{W13}=1/8$ MiB，而
$M\ge224$ 的最优是 $1/2$ MiB，$M=256$ 处损失 $23\%$（`1T` 上 $61\%$）。
$1/2$ 与 $1/8$ 的交叉点由 $M=200$（$1/8$ 领先 $3.9\%$）与 $M=224$（$1/2$ 领先
$1.6\%$）内插得 $M\approx217$。production 默认升级到
`amazon_c5_192c_tp4_f512_v4`，把 band 拆成 `144--215` 与 `216--287`：边界取 216 即
整数个 M12 panel，接缝处最多损失 $0.2\%$；`144--215` 沿用 V1 的原值（含 `8T`
override），新 band `216--287` 取 $(1/2,1/8)$ 且不设 override；`288--575` 不动。

catalog 里落在 `216--287` 的 expert 极少——只有 `dsv4-real-2048-seq70` 的 3 个
task（223 个中的 3 个，约 2% route），其余 preset 为 0。因此五个 preset 的端到端
差异全部在 $\pm0.78\%$（同期 `legacy` 变体自身摆动 $-1.28\%$--$+0.34\%$）、所选
shape 全部不变，收益是潜在的。

该门限已确认不是 paging 假象。统一 page policy（`FUSED_CPP_PAGES=small|thp|hugetlb`）
可以对包含 packed-A staging 在内的每个 buffer 切换页大小，在 $M\in\{192,224,256\}$ 上
三种页大小给出**逐格相同的 argmax**、同样的 $M=192\to224$ 台阶，绝对值差 $\le0.6\%$。
因此这是 cache 容量效应而非地址转换效应，A 侧的 TLB 混淆由此以实测排除而非假设排除。

数据在 `thresh_m{120,160,176,192,200,224,240,256,300,320}.json`、
`thresh_m256_t{1,2,8}.json`、`pages_{thp,hugetlb,small}_m{192,224,256}.json`
与 `v4_{preset}.json`。

#### 9.27 历史结论：为什么旧模型不能直接退出 split/no-split 候选维度

> 当前状态（2026-08-09）：本节针对当时仍有“未覆盖 task 继承全局 pair”的
> V3 模型。v0.88 先让所有 task 在 team width 确定后经过 $g_\theta(M,t)$ lowering，
> 再删除全局 pair/profile variant，因此不再依赖这里的完整 legacy pair 前提。
> 下文保留为迁移决策历史，不描述当前代码。

8.2.5 指出 split/no-split 是 $g_{W13}\in\{4,8\}$ MiB 的布尔别名，由此自然会问它能否
在 $\omega$ band 覆盖足够广时退出 planner 的候选维度。实测结论是**不能**，因为
band 永远不会覆盖全部 $(M,t)$：

| 继承原因 | 是否可以补 band |
| :--- | :--- |
| $M\le12$ | **不应该**。$P=1$ 时没有 packed-B 复用可保护，实测窗口效应仅 $\pm0.7\%$ |
| $M>575$ | 可以，但需新标定；且 $M=2040$ 的 $\omega^\ast\approx1$ MiB 与 legacy 在 `4T` 上给出的值一致 |
| $t>8$ | **不应该**。9.26 实测 $\omega_{\text{legacy}}=4\ \mathrm{MiB}/t$ 在 `16T`/`32T` 已接近最优，且 $\omega$ 在那里不可迁移 |

V3 在 $(M,t)$ 网格上的覆盖率为 $52\%$。production preset 上仍走 operator-wide 几何
的 task 占比为：`moe256-uniform` 与 `moe256-active-set-128` 为 $0\%$，
`moe256-tiered-hotspot` $6\%$，`dsv4-real-2048-seq70` $35\%$，而
`moe256-long-short-bimodal` 为 **$100\%$**——它的 route 只有 $\{12,2040\}$、宽度只有
$\{1,16\}$，每一格都落在上表三种原因里。对该 workload，split/no-split 是**唯一**的
窗口控制。

因此在**当时实现**中，operator-wide 几何还不是可以单独退役的兼容层；
`policy_variants()` 的完整 legacy pair 前置条件用于保证两种活跃几何都有标定数据。
v0.88 删除继承和联合搜索后，这个实现约束已经消失。

#### 9.29 team 分割几何（历史 `kM` / `kN` policy）与 $\omega$ 的适用域

> 当前状态（2026-08-09）：production `plan_team_gemm_split` 的
> `default_split_selector` 对 W13/W2 均固定返回 `kN`；`choose_moe_gemm_split`
> 的旧门限只保留给显式 benchmark/诊断，不再决定 fused-expert 主路径。所以下文
> `kM` 交界表描述的是旧运行时和旧测量的解释边界，不能用于当前解析
> stage-window policy。当前 policy 的所有候选均满足 $\omega=g_s/t$ 的 `kN`
> 语义；未来若重新启用 `kM`，必须先给解析 mapper 增加 axis-aware demand，不能
> 复用本次 holdout 结论。

$\omega=g_s/t$ 被称为「每线程窗口」，但这个物理含义依赖 team 沿哪个轴分割。
`csrc/moe/arm/common/fused_moe_bf16_tiled.cpp` 的 `choose_moe_gemm_split` 按
$(stage, M, t)$ 选择，weight-window 循环则**无论如何都只切 N 轴**
（`make_weight_window_plan` / `for_each_weight_window` 交出 `n_range`）。M 轴只以
micro-kernel 的 8/12 行寄存器块与 team 分割轴两种形式出现，**不存在 cache 级的
M 分块循环**。

两种几何下每线程看到的量不同：

| | `kN`（N 轴分给线程） | `kM`（M 轴分给线程） |
| :--- | ---: | ---: |
| 每线程 A | $2MK_s$（全部 $M$ 行） | $2(M/t)K_s$ |
| 每线程 B 窗口 | $g_s/t=\omega$ | $g_s=t\omega$（全队共用同一窗口） |
| 每 range 全队 A | $t\cdot 2MK_s$ | $2MK_s$ |

因此 **$\omega$ 只在 `kN` 下等于每线程 B 占用**；`kM` 下每线程占用是 $g_s$ 本身，
比 $\omega$ 大 $t$ 倍。而 policy 无条件按 $g_s=t\omega$ 下放（`_lower` 经
`range_bytes_for_worker_window`），不区分几何。

W13 的分割规则与 band 的交界：

| 规则 | 触发 | 影响 |
| :--- | :--- | :--- |
| $t=1$ | 恒为 `kN` | — |
| $t=2$ 且 $M\le32$ | `kM` | **band `13--48` 在 $t=2$ 处被切开**（13--32 为 `kM`，33--48 为 `kN`） |
| $t=4$ 且 $M\ge1024$ | `kM` | 所有 band 上界 575，不触发 |
| $t=8$ 且 $M\ge512$ | `kM` | **band `288--575` 在 $t=8$ 处被切开**（512--575 为 `kM`） |
| $t>8$ 且 $M>N$ | `kM` | 短 route 侧 $M\ll N=2F$，不触发 |

W2 的门限高得多（$t=8$ 需 $M\ge4096$、$t=4$ 需 $M\ge8192$），故 catalog 内
**W2 恒为 `kN`**。两个 stage 因此可以在同一次调用里走不同几何：$M=2040$、
`24x4T` 下 W13 为 `kM` 而 W2 为 `kN`。

这修正了 8.2.5 与 9.28 的两处叙述。8.2.5 的 $M=2040$、`24x4T` 数据 W13 走 `kM`，
每线程 A 为 $3.98$ MiB（$A/L2=1.99$）而非按 `kN` 读法的 $16.7$ MB
（$A/L2=7.97$）——**凡引用该点 $A/L2$ 做外推处需按 1.99 重算**。该点的 $16.7$ MB
仍然正确，但它是**每 range 全队**量而不是每线程量。9.28 的门限推导以
「每线程读全部 A 行」为前提，其全部测点为 $M\le320$、$t\in\{1,2,4,8\}$，除
$M\le32\ \&\ t=2$ 外均为 `kN`，故在该节适用域内成立。

9.25 记录的 $M=28$、$t=2$ 格与其他宽度差数个百分点，现在有了机制解释：该格是
`kM`，其每线程 B 占用为 $2\omega$ 而非 $\omega$，与同 band 其他宽度不是同一物理量。
校准值本身是实测所得、描述的是真实行为，因此不因本节而改动；**改动的是解释，以及
任何基于 $\omega$ 跨宽度外推的合法性**——跨越上表交界的外推无效。

未验证：`kM` 区间是否存在自己的 $\omega^\ast(M)$ 规律（现有标定未按几何分层）；
以及把 $\omega$ 在 `kM` 下重定义为 $g_s$ 后重新标定 `13--48@t=2` 与 `288--575@t=8`
两格是否会改变数值。两者都需要新测量。

#### 9.30 band 表的实际覆盖率与 band 内梯度

把 catalog 的九个 preset 全部经 production planner 规划，按每个 task 的
expert route 数与所选 team 宽度归格，`amazon_c5_192c_tp4_f512_v4` 的 24 个格中
**只有 6 个被触及**：

| band | 1T | 2T | 4T | 8T |
| ---: | :---: | :---: | :---: | :--- |
| 13--48 | — | — | — | `dsv4`:120, `uniform`:256 |
| 49--95 | — | — | — | — |
| 96--143 | — | — | `active-set-128`:128 | `tiered-hotspot`:48 |
| 144--215 | — | — | — | `dsv4`:15, `active-set-64`:64 |
| 216--287 | — | — | — | `dsv4`:3 |
| 288--575 | — | — | — | `dsv4`:7, `active-set-32`:32, `tiered-hotspot`:12 |

全部 1T/2T 格为空，4T 仅 `96--143` 一格。另有 285 个 task 落在表外继承
operator-wide 窗口，其中 `moe256-long-short-bimodal` 的 179 个（route
$\{12,2040\}$）全部继承，与 9.27 的结论一致。

这直接解释了三处 latent gain：9.26 补的 `49--95` 窄格全空、9.28 补的 `216--287`
只有 `dsv4` 的 3 个 task、9.26 保留 16T/32T 继承。三次 A/B 都只能报噪声，原因不是
标定错而是**没有 preset 会排到那里**。因此补格子之前应先查覆盖，而不是等 A/B
不动之后再解释。

唯一活跃的 4T 格还暴露了 A/B 的结构性盲区：`moe256-active-set-128` 在 A/B 集合内，
但两臂分别是 `legacy` 的 $4\ \mathrm{MiB}/t=1$ MiB 与 `policy` 的 $0.0625$ MiB，而
胜过两者的 $0.125$ MiB **不在任何一臂中**，重复多少次都测不到。

`96--143` 用单一 $\omega$ 覆盖 48 个 route，而最优 tile 数随 $M$ 单调上升。4T、
$g_{W2}$ 固定 $0.5$ MiB、192 同构 expert，八点 $M$ 网格（墙钟 ms）：

| $M$ | $A/L2$ | 1 tile | 2 tile | 4 tile | 最优 | 与次优差 |
| ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 28 | 0.109 | 8.075 | 8.077 | 8.048 | (4) | $0.33\%$ |
| 40 | 0.156 | 8.744 | 8.733 | 8.682 | (4) | $0.59\%$ |
| 56 | 0.219 | **9.362** | 9.452 | 9.529 | 1 | $0.96\%$ |
| 72 | 0.281 | **9.977** | 10.202 | 10.585 | 1 | $2.26\%$ |
| 84 | 0.328 | **10.759** | 10.892 | 11.383 | 1 | $1.24\%$ |
| 96 | 0.375 | **11.687** | 11.695 | 12.302 | 1 | $0.07\%$ |
| 108 | 0.422 | 12.979 | **12.736** | 13.322 | 2 | $1.91\%$ |
| 120 | 0.469 | 14.170 | **13.874** | 14.398 | 2 | $2.13\%$ |

两个结论。**$M\le40$ 时窗口无关**：窗口变化 4 倍只动 $0.33$--$0.59\%$，在约 $0.5\%$
的重复性之内，故 `13--48` band 的 $0.25$ MiB 是三个不可区分选项之一而非有意义的最优。
**$M\ge56$ 起最优为 1 tile，在 $M\approx100$ 处跨到 2 tile**：合上 9.28 在
$M\ge224$ 的 $1/2$ MiB，阶梯是 1、2、8 tile/线程，**单调无下凹**。此前把这些点的一个
子集读成非单调，是因为混用了不同的 $g_{W2}$。

由此得到一项二阶耦合：**W2 窗口会改变 W13 曲线的形状**。$M=28$ 上 $g_{W2}=0.25$ MiB
时 W13 轴有 $2.0\%$ 梯度（8.240/8.152/8.077），$g_{W2}=0.5$ MiB 时变平
（8.075/8.077/8.048）。9.25 以自身极差认定 W2 为弱轴，该结论不变，但两轴的**形状**
并不独立。

对 shipped 表，这改变 `96--143@4T` 的读法而不改变决策：台阶落在 band 内的
$M\approx100$，而 `moe256-active-set-128` 正好在 $route=96$，两个窗口相差 $0.07\%$；
route $108$--$143$ 在 4T 上相对 shipped 值约付 $2\%$。另两个压在 1-tile 下界的格子
实测**生产值正确**：`49-95@4T`（$M=72$）1 tile 优 $2.0\%$，`96-143@2T`（$M=120$）
1 tile 优 $0.27\%$。故 `96-143@4T` 是单一的 band 边界个案，不是下界处的系统性偏小。

`2T` 那格另有意义：同一 $M=120$，$t=2$ 最优 1 tile 而 $t=4$ 最优 2 tile。8.2.5 的宽度
不变性是"同 $\omega$ 下极差 $3.0\%$--$5.6\%$"，该表述仍成立，但**argmax 本身在此处
不是宽度无关的**。同 $\omega$ 下两者 $R$ 差 2 倍，是明显嫌疑，未验证。

#### 9.31 窗口的残余代价不在任何已测计数器中

$M=72$ 上 1 tile 与 4 tile 墙钟差 $5.4\%$，而 $R$ 变化 4 倍时全部计数器持平
（每迭代增量）：

| $M$ | $\omega$ | $R$ | 墙钟 ms | instructions | l1d access | l1d refill | l2d refill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 72 | 0.0625 | 32 | **9.954** | 9.57G | 3.10G | 0.05G | 52.6M |
| 72 | 0.125 | 16 | 10.113 | 9.76G | 3.13G | 0.05G | 52.1M |
| 72 | 0.25 | 8 | 10.495 | 9.74G | 3.14G | 0.05G | 52.6M |
| 120 | 0.0625 | 32 | 14.137 | 15.38G | 5.00G | 0.10G | 66.0M |
| 120 | 0.125 | 16 | **13.858** | 15.41G | 5.06G | 0.06G | 62.5M |
| 120 | 0.25 | 8 | 14.516 | 15.44G | 5.06G | 0.07G | 63.2M |

$M=72$ 处 L1 refill 恒为 0.05G、L2 refill 变化 $1\%$，指令变化 $2\%$，而墙钟变化
$5.4\%$。$M=120$ 处 L2 refill 确实跟随最优（66.0→62.5M），但 $6\%$ 的填充摆动不足以
单独解释 $2.1\%$ 的墙钟摆动。

这否证了此前所有候选：**非 L2 容量**（该区间 $A+\omega$ 只占私有 L2 约 $30\%$）、
**非 A 重扫流量**与**非 B 重读流量**（两个 refill 对 $R$ 的 4 倍变化无响应）、
**非指令数**（故非 micro-kernel 摊销）、**非 barrier**（9.32 显示合法区恒为 $1.2\%$）、
**非并发干扰**（big expert 由 28 降到 1，墙钟差 $1\%$）。

五个计数器持平而墙钟差 $5.4\%$，指向**延迟而非体量**：range 更少意味着 N 方向连续
扫描更长，可以改变缺失的重叠程度而不改变其数量。这与 9.31 之前的页大小结果同类
（字节数相同、墙钟差 $13\%$、机制为预取连续性）。可判别的测量是把
`l2d_cache_refill` 拆成需求与预取，或看内存停顿的分布而非总量。

#### 9.32 barrier 代价是线程饥饿的阶跃函数，与 range 数无关

stage window 以整个 packed-B tile 量化，$K\cdot\text{n\_tile}\cdot2$，W13 为
$64$ KiB、W2 为 $8$ KiB，且 `make_weight_window_plan` 用
`max(1, weight_window_bytes / bytes_per_tile)` 把 range 夹到至少一个 tile。tile 不可
再分：packed 布局以 tile 连续，micro-kernel 输出块宽 `n_tile`。`kN` 下这给出每线程
窗口的下界**恰为一个 tile 且与 $t$ 无关**——range 内 tile 数少于线程数时，多余线程
拿到零工作量。

`ThreadBarrier::wait()` 的采样占比（$M=120$，4T）：

| tile/range | 空闲线程 | $R$ | barrier 占比 |
| ---: | ---: | ---: | ---: |
| 1 | 3/4 | 128 | $\mathbf{41.78\%}$ |
| 2 | 2/4 | 64 | $\mathbf{21.86\%}$ |
| 4 | 0 | 32 | $1.19\%$ |
| 8 | 0 | 16 | $1.19\%$ |
| 16 | 0 | 8 | $1.57\%$ |
| 32 | 0 | 4 | $1.28\%$ |

占比跟随空闲线程比例，其余情况持平：后四行 $R$ 变化 8 倍而占比不动。所以 barrier
**不是随 range 累积的代价，而是窗口饿死线程时才触发的阶跃**。

两点对读 profile 很重要。此处的等待是**自旋**，以高 IPC 退休指令（$4.76$ 对最优点的
$3.62$），计入 `instructions` 而**不计入** `stall_backend`；因此越界的窗口看起来像
"多做了工作"而不是"多等了"。以及 shipped 表从不越界：最小 $\omega_{W13}$ 为
$0.0625$ MiB 即恰好一个 tile，而 $\omega_{W2}$ 距其自身 $8$ KiB 下界还有 8 倍。

旧运行时由此有一个结构性缺口但生产不可达：tile 数少于线程数的 range 可沿 M 切以
用满线程，但旧 `choose_moe_gemm_split` 只按 $(stage,M,t)$ 选轴、从不查窗口。当前
production 已固定 `kN`，而 8.3 的解析候选显式要求每 range 至少
$\min(t,q_s)$ 个 tile；因此会饿死线程的点在进入目标函数前就被排除。

#### 9.33 解析生成 stage-window policy 的两机 holdout

2026-08-09 将 8.3 的确定性 $g_\theta(M,t)$ 从经验 band 扩展为解析 backend 可直接
生成的 policy。候选、约束和增量目标见 8.3；实现为
`cost_model/analytic_stage_window_policy.py`。`PlannedMoE` 对 analytic model 优先
绑定该 policy，对 empirical model 继续使用精确 profile 的 V4 表或原 fallback。
shape、tail-pool、kernel variant 和剪枝集合逐项不变。

holdout 固定 `H=4096,F=512`、独立 expert weights、同一 lane DAG，只改变 W13/W2
window。route 为 `28,72,120,216,320,768,2040`，width 为 `1,2,4,8`。每个
$(M,t)$ 测 analytical、inherited、完整 W13/W2 单轴及 analytical 邻域 $3\times3$
交叉点；该 coordinate oracle 不是完整二维笛卡尔积，故 regret 是对完整 legal space
的下界。候选按 round 随机交错，避免先连续测完一个窗口造成时间漂移偏置。

| 机器/口径 | median regret | P90 | max | $\le2\%$ | $\le5\%$ | 相对 inherited median gain |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AmazonC5192Cores NUMA0，96 experts，11 rounds | 1.63% | 6.57% | 11.32% | 14/28 | 21/28 | 53.83% |
| AmazonECSV1 8C，32 experts，15 rounds，raw median | 1.39% | 3.51% | 15.22% | 18/28 | 27/28 | 5.59% |
| 同一 8C 数据，p10 单边噪声敏感性 | 1.30% | 1.93% | 2.78% | 26/28 | 28/28 | 5.43% |

192C 的候选墙钟 `p90/p10` 中位仅 $1.44\%$，最大误差在 p10 口径仍为 $11.22\%$，
故失败是真实模型误差：$M=2040/768@1T$ 把 W13 owner window 选成 512 KiB，coordinate
oracle 为 1024 KiB；$M=216@1T/2T$ 和 $M=72@8T$ 又在相反方向错过转折。候选排序
Spearman $\rho$ 的组间中位为 0.853、28 组中 26 组为正，说明 broad trend 已捕获，
但 M=216 transition、长 route A/range balance 和两 stage 二阶耦合尚未闭合。

8C 的 `p90/p10` 中位为 $19.56\%$、P90 为 $101.50\%$，后台 remote-agent 造成明显
单边抢占；raw 最大点 $M=2040@4T$ 的 analytical median/p10 为
`681.50/578.28 ms`，oracle 为 `591.46/576.39 ms`。所以该机只能说明可迁移方向，
不能用 raw median 判严格 5% gate。其 cache/service curve 均为本机测量，但
packed-B retention 暂用 192C 的 $1/8$-L2 prior，并在 profile provenance 标记为
non-local；仍需本机 multi-team refill probe。

两机另各跑 `M=12,T=4`、21 个 interleaved rounds。inherited owner window 均为
1024/1024 KiB；192C 缩到最小 W13/W2 后由 3.2409 ms 变为 3.3742/3.2870 ms，8C
由 4.4063 ms 变为 4.5329/4.5638 ms。中间 W2 点非单调，不据此拟合全局
`range_fixed_ns`；$M\le12$ 保持继承。

v1 结论（已被下方 v2 修正取代）：解析 backend 直接生成 policy 已可用作
portable/shadow execution policy，但 clean 192C 的 max regret 未通过 5% gate，
不能替换 empirical production V4。
完整数据、噪声说明及 V4 对照见
`optimizations/fused_moe_sve/results/analytic_stage_window_policy_holdout_20260809.md`。

**Policy v2 修正。** v1 的两个稳定误差来自同一流量抽象：它把短 repeated-scan
probe 的残余 $g_2$ miss 乘到全部 $P-1$ 个 panel，导致长 route 的小 miss 无界
累积；同时 A 只走 effective/physical smoothstep，没有表达“完整 A 能否跨顺序 N
range 驻留”的物理二态。v2 按 8.2.4 修正为：

1. B 的 transient reuse 数由
   $L_A=\max(\lfloor C^A_{2,eff}/A_p\rfloor,1)$ 给出，之后按
   $U_j+A_p\le C_2$ 判 resident/streaming steady state；
2. A 按 $U_j+A\le C_2-A_p$ 判跨 range 驻留，给在途 load/prefetch 留一个 panel；
3. 若 $J_s$ 差异小于 $\epsilon_{rel}T_{xfer}$，不再相信 sub-microsecond 的顺序，
   而在等价集内取最接近 $\max(C_{L1D},2b_s)$ 的自然 owner window。

这三个量都来自 cache、panel、packed tile 几何或 machine-level uncertainty；没有
新增 route 阈值、时延格点或 planner 搜索变量。相同 grid、coordinate oracle 和
interleaved 协议重新测得：

| 机器/口径 | median regret | P90 | max | $\le2\%$ | $\le5\%$ | 相对 inherited median gain |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AmazonC5192Cores NUMA0，median | 1.50% | 2.98% | 3.38% | 17/28 | 28/28 | 56.13% |
| 同一 192C 数据，p10 | 1.64% | 2.87% | 3.46% | 16/28 | 28/28 | 56.19% |
| AmazonECSV1 8C，raw median | 2.88% | 6.05% | 18.41% | 10/28 | 22/28 | 3.04% |
| 同一 8C 数据，p10 | 2.01% | 2.61% | 4.21% | 14/28 | 28/28 | 4.58% |

192C candidate `p90/p10` spread 的 median/P90/max 为
`1.46/3.20/8.22%`，v2 将 v1 的 `1.63/6.57/11.32%` 降到
`1.50/2.98/3.38%`，clean 5% stage-window gate 已通过；median rank $\rho$
由 0.853 升到 0.903。最大残差为 $M=216@2T$ 的 3.38%，其次是
$M=2040@1T/2T$ 的 3.27/3.10%，现在主要是 W2/双 stage 二阶耦合，而不再是
W13 transition 或长 route A-scan 的一阶错误。

8C 的 candidate spread median/P90 仍为 `28.47/99.66%`，raw 18.41% 最大值
来自单边抢占；p10 最大 4.21%。该机 packed-B retention 仍是 transferred prior，
所以这只是方向性 portability pass，不能替代本地 multi-team probe。解析 backend
保持 v2 默认；empirical V4 仍为 production default/fallback，直到完整 contention、
distributed lifetime 和两机本地校准 gate 均通过。完整 v2 记录见
`optimizations/fused_moe_sve/results/analytic_stage_window_policy_v2_holdout_20260809.md`。

**Policy v3 range 统一。** 迁移的第一步只修改解析候选，不改变 empirical
production profile、Plan V2 ABI 或 native kernel。W13 的旧 split/no-split 分别
归一为 $R=2/R=1$ 端点，两点与 inherited、解析自然窗口一起进入同一目标函数；
policy identity 记录 inherited 的两个 stage range，而不再记录 split chunk 数。
$M\le12$ 由单 panel 支配关系固定选择 $R_{W13}=R_{W2}=1$。holdout 工具默认加入
$M=12$、`1/2/4/8T`，并给 W13 的两个端点写入独立标签，保证后续 range ABI 迁移前后
可做逐点 shadow 对照。当前提交只增加候选覆盖和单元验证，**不复用 v2 的 3.38% 数字
声称 v3 实机 gate 已通过**；完整实机数据在 runtime/profile range 化后统一刷新。

以上 9.33 全节是 range/window 运行时仍存在时的历史验证，不再定义当前生产模型。

#### 9.34 full-N team-stripe 迁移与验收

2026-08-09 的 v0.89 删除了 production weight split/range/window。每个 expert 的
W13 和 W2 各执行一次完整 packed-N domain；team 宽度 $t$ 是唯一的 N-owner 窗口
控制量。令 $q_s=N_s/\nu$ 为 stage $s$ 的 N tile 数，则最大 worker stripe 为

$$
u_s(t)=2K_s\nu\left\lceil\frac{q_s}{t}\right\rceil,
$$

而完整 stage 权重分别固定为 $B_{13}=4HF$、$B_2=2HF$。这里的
`task_range_granularities` 仍只表示 route/M slicing，不是 weight-N split。

迁移同步删除 Python/C++ public ABI 的 `w13_ranges/w2_ranges`、Plan V2 的
`task_w13_ranges/task_w2_ranges`、ARM/x86 production range 循环、native planner
bridge、解析 stage-window policy、profile identity 和默认 benchmark 控制。旧
split profile 从 active catalog 删除；catalog 对非 `full_n_team_stripes` 几何和
历史非单位 range metadata fail closed。显式 vLLM staged N-task baseline 与独立
microbenchmark range 开关只保留为实验/历史对照，不可进入 production calibration。

验收结果：

- macOS ARM64 stub build 通过；核心 planner/model/runtime 回归为
  `174 passed, 101 skipped`。全仓额外失败来自本机不可用的 KleidiAI 和既有 MLA
  mock，与本次 MoE 变更无关。
- `AmazonC5192Cores` 重新编译真实 AArch64 SVE/JIT/asm 扩展成功；在 NUMA0
  `0-95` 核运行同组测试为 `274 passed, 1 skipped`。
- 同机 HugeTLB TP4 smoke（$H=4096,F=512,M=12$）输出
  `stage_geometry=full_n_team_stripes`，记录 W13/W2 stage 为 `8/4 MiB`；1T/4T
  isolated 每 expert median 为 `0.579/0.185 ms`，单个 `4T` contention group 为
  `0.187 ms`、derate `1.010`。该三轮 smoke 只验证执行与 metadata 契约，不作为
  新的 production calibration table。

#### 9.35 窗口作为一等参数：$(t,\omega_{13},\omega_2)$

v0.89 的 full-N team stripe 是 $R=1$ 端点，不是唯一几何。把每线程 owner 窗口
$\omega_s$（以整 N tile 计）恢复为一等参数后，一个 stage 的计算模式由
$(t,\omega_s)$ 唯一确定：

$$
\text{range}_s = t\,\omega_s,\qquad
R_s = \left\lceil \frac{q_s}{t\,\omega_s} \right\rceil,\qquad
\omega_s = \left\lceil q_s/t \right\rceil \iff R_s = 1
$$

窗口 $i$ 覆盖 tile $[i\cdot\text{range}_s,\ \min((i+1)\text{range}_s,\ q_s))$，尾窗口
可短，窗口内按 `split_evenly` 分给 $t$ 个线程。**$R=1$ 即 9.34 的 full-N team
stripe**，故新参数化是旧几何的超集，默认逐位一致。

三条结构性质。**纯调度旋钮**：每个 (M panel, window) 对被访问恰好一次、各线程写
disjoint C、融合 W13 无 K 分块，故任何合法 $\omega$ 输出逐位相同。**饥饿边界可
判定**：尾窗口 $r=q_s \bmod \text{range}_s$，若 $t\mid q_s$ 则 $r$ 必为 $t$ 的倍数、
永不饿死线程；TP4 下 $q_{13}=128$、$q_2=512$ 使 $t\in\{1,2,4,8,16,32\}$ 全部安全。
**下界为一个 tile**：packed B 以 tile 连续、micro-kernel 输出块宽 $\nu$，故
$\omega_s\ge1$ 且与 $t$ 无关。

ABI 传 tile 数而非字节。字节口径的 $\lfloor b/b_s\rfloor$ 换算是多对一，无法反推唯一
模式，故 `FullStageGeometry.window_tiles_from_bytes` 是唯一允许换算的地方，Plan V2
的 `task_w13_window_tiles`/`task_w2_window_tiles` 只接受 tile 数，$0$ 表示全条带。
两者与 route/M slicing 的 `task_range_granularities` **正交**：切 route 改 $M$，不改
worker 拥有哪些权重列。

性能中性的 stripe 级 `bulk_m` 已从运行时和 JIT 生成器移除；它原本假设一次调用覆盖
worker 的整条 stripe，与窗口下每个 window 独立调用的执行模式不兼容。
`first_panel_prefetch` 同样因 E2E 无稳定收益且高并发、长 route 可回退而退役；两者仅在
Git 历史与实验报告中保留。

**代价与收益**（`results/amazon_192c_stage_window_tiles_20260810.md`）。全条带在窄
宽度上代价很大：$t=4$ 时每 worker 2 MiB，96 核并发使 192 MiB packed B 同时活着对
96 MiB L3，实测比最优窗口慢 $75\%$--$104\%$（$M=56$--$120$）；$t=8$ 慢
$23\%$--$31\%$；$t=16$ 在 $M\le120$ 慢 $4\%$--$15\%$。但 $t=16$ 且 route $\ge144$
起全条带最优，强制 1 tile 在 $M=384$ 慢 $80\%$——大 $M$ 区 $A$ 不再常驻，需要大窗口
摊销其重扫，与 9.28 的门限方向一致。$t=32$ 的 stripe 已只有 4 tile，所有窗口在
$0.6\%$ 噪声内。

policy 是 route band 表 + per-width override，在**已选定**的 $(M,t)$ 上确定性读表，
**不增加 planner 搜索维度**。未覆盖的 route/width 返回全条带，故报
`stage_geometry=full_n_team_stripes` 且行为与迁移前一致，旧 profile 继续有效；只有
$R>1$ 时报 `windowed_team_stripes`。planner 生成的 plan 上端到端 A/B：
`moe256-uniform` $+12.4\%$、`dsv4-real-2048-seq70` $+10.6\%$，
`moe256-tiered-hotspot` $+1.5\%$ 与 `moe256-active-set-128` $-0.65\%$ 在
$1.3\%$--$1.7\%$ 噪声底内（噪声底由"两臂 plan 完全相同"测得）。

未验证：`moe256-active-set-128` 的 52 个加窗 task 正落在孤立实测 $+9.2\%$ 的格子却
端到端不动，推测因其余 76 个 task 在 $t=32$（policy 不动）决定 makespan，未测。
$t=16$ 的项只标定了 W13 轴，其 W2 项取全条带。宽度 $1,2,3,6,12$ 沿用 V4 或未覆盖。

#### 9.36 解析 stage-window v6：选择问题闭合，绝对时间分解仍有边界

9.33 的 `1.50/2.98/3.38%` 是旧 range ABI 上的 coordinate oracle：它证明 v2 在已测
坐标附近**选择接近最优**，但既不是完整 W13 x W2 笛卡尔积，也不能证明每个候选的
绝对时间被正确解释。当前 tile-window runtime 因此重新定义闭合门槛：在预先声明的
192C TP4 转折域上跑全部合法二阶组合，要求

$$
\max_{(M,t)\in\mathcal H}
\frac{T(\widehat w_{13},\widehat w_2)-
      \min_{w_{13},w_2}T(w_{13},w_2)}
     {\min_{w_{13},w_2}T(w_{13},w_2)}
\le 5\%.
$$

这里 $\mathcal H=\{72,216,384\}\times\{4,16\}$，覆盖短 route、$A/L2$ 转折和
长 route，以及窄/宽 team；它是当前高价值 window 选择域，不代表所有机器、shape 或
完整 planner 已闭合。

**精确执行几何。** 对 stage $s$，令 packed-N tile 数 $q_s=N_s/\nu$、单 tile 字节
$b_s=2K_s\nu$、每线程窗口为 $w_s$ tiles。则

$$
g_s=t w_s,\qquad
R_s=\left\lceil\frac{q_s}{g_s}\right\rceil,
$$

尾窗口由 `split_evenly` 分配；$w_s=\lceil q_s/t\rceil$ 是 full-stripe 端点。候选只
保留 full stripe 与满足 $w_sb_s\ge C_{L1D}$ 的二次幂 owner window，并剪掉会让任一
线程在尾窗口无 tile 的点。$M\le12$ 只有一个 M panel、没有 B 重用可保护，按支配关系
固定 full stripe。这个规则是已选 $(M,t)$ 上的 deterministic post-policy，不增加 shape
搜索变量；production 仍使用现有 band policy，v6 只作 shadow。

**解析需求。** 令 $P_s=|\mathcal P_s(M)|$ 为 exact-M mapper 生成的 panel 数，
$A_s$ 为完整 packed A，$A_{p,s}$ 为最大单 panel。每个 window 分别计算：

1. 必需 packed-B 首读 $n_jb_s$；
2. effective L2 周转期内的 transient B refill，以及其后的 resident/streaming 状态；
3. 当 $A_s+U_j$ 超过为在途 panel 留出 headroom 后的 L2 容量时，跨 window 的 A refill；
4. C 的流式写入。

task-local transfer 保留 A/B/C 全部流量。rank 级容量附加项只给**可重用 B** 记 resident
budget：若 $h=\lfloor T_{rank}/t\rfloor$ 个完整 team 同时工作，则

$$
G_{B,j}=h\,n_jb_s,
$$

LLC spill 增量只乘 $\max(B^{L2}_j-n_jb_s,0)$。A 与 C 仍消耗带宽，但二者是一次性流，
不能作为同等大小的 B 驻留集合再次扣 LLC 容量。旧实现把三者相加，系统性高估大 window
的共享 cache 代价。

**range restart。** 纯 GEMM/no-store probe 把额外 N range 的代价测成约 $6.70$ ns，
但 fused W13 真实执行 SiLU、gate multiply 与 BF16 packC，独立 L1-hot probe 为
$45.27$ ns。因此使用 stage-specific 项

$$
T_{restart,s}=P_s(R_s-1)\delta_s,
$$

当前 $\delta_{13}=45.27$ ns，$\delta_2=6.70$ ns；W2 是 pure no-store proxy，仍是显式
校准边界。两条曲线分别用 256 warmups、8192 timed calls、7 组随机 range 顺序拟合，
最大拟合误差为 $0.087\%/0.180\%$。同步 barrier 假设被独立否证：$M=216,t=4$ 加 team
critical-path 同步后 W13 仍选择 8 tiles，同步税仅 $1.9\%$--$3.9\%$，不足以改变最优点，
所以模型不加入不可识别的 barrier 拟合项。

最终 stage 目标为

$$
J_s=\sum_j\left[T_{core,j}+T_{epi,j}+
\max(T_{L2,j},T_{LLC,j},T_{DRAM,j})\right]
+T_{restart,s}+T_{sharedB,s}.
$$

若候选差小于 $\epsilon_{rel}(T_{xfer}+T_{sharedB})$，模型承认其在机器不确定性内等价，
再用纯 cache 几何确定 tie-break：

$$
U_{pref,s}=\max\left(C_{L1D},2b_s,
\sqrt{C_{L1D}C^{B,eff}_{L2}}\right).
$$

192C 上 $C_{L1D}=64$ KiB、$C^{B,eff}_{L2}=256$ KiB，故偏好 128 KiB；没有新增
route、thread 或 latency table。需要校准的只有硬件/实现 service：GEMM、L2/LLC/DRAM、
cache 容量、B retention 三锚点、相对不确定性和两个 $\delta_s$；panel 数、流量、候选集、
cohort 大小均由公式生成。

**完整 Cartesian holdout。** `AmazonC5192Cores` NUMA0 `0-95`，$H=4096,F=512$，
96 份不同 expert 权重，32 MiB HugeTLB；每组 2 warmups + 15 个随机交错 rounds：

| $M$ | $t$ | v6 W13/W2 tiles | oracle tiles | regret |
| ---: | ---: | :--- | :--- | ---: |
| 72 | 4 | 2 / 16 | 1 / 8 | 4.40% |
| 72 | 16 | 2 / 16 | 1 / 8 | 1.74% |
| 216 | 4 | 8 / 16 | 2 / 16 | 2.51% |
| 216 | 16 | 8 / 16 | 8 / 16 | 0.00% |
| 384 | 4 | 8 / 16 | 8 / 16 | 0.00% |
| 384 | 16 | 8 / 16 | 8 / 32 | 0.47% |

median/线性插值 P90/max regret 为 `1.10/3.45/4.40%`，4/6 在 2% 内、6/6 通过 5% gate；候选
Spearman $\rho$ 中位为 0.869。因此**当前域内的 window 选择问题闭合**：解析公式无需
$(M,t)$ 时间表即可选择 5% 内的 runtime pair。

边界同样明确。paired-round 的 W13/W2 可加性残差在极端未选点上最高仍为 17.69%，且
$M=72,t=16$ 的全排序 $\rho=0.098$，虽然 selected regret 只有 1.74%。所以本结论不是
“模型精确解释了每一个时间”，更不是 production default 的采用证据；它只证明当前域
内的决策质量。8C 的本机 B-retention、其他 route/width、混合 workload 和完整 planner
仍需独立 holdout，production policy 与 planner 剪枝保持不变。完整协议和反例见
`optimizations/fused_moe_sve/results/analytic_stage_window_policy_v6_closure_20260811.md`。

#### 9.37 大/小 route 双资源区的 DSV4 首轮验证

统一 shape 把所有 active expert 放入同一种宽度的 lane。对 DSV4 这类长尾
histogram，这会让短 route 用过宽 team，却又只保留少量并发 cold-B stream。第一版
实验引入一个不改变 kernel、window policy 或 Plan V2 ABI 的静态分区候选。给定 route
阈值 $\theta$：

$$
\mathcal J_L=\{i:M_i>\theta\},\qquad
\mathcal J_S=\{i:0<M_i\le\theta\}.
$$

把单 NUMA 的 $T$ 个核心分成连续且互不重叠的 $C_L,C_S$ 两区，满足
$C_L+C_S=T$。两区分别使用已校准宽度 $t_L,t_S$，并要求
$t_L\mid C_L,t_S\mid C_S$。lane 数为 $h_x=C_x/t_x$；各区内部按
$I_i(t_x)$ 降序做 LPT，形成独立依赖链，两区从同一调用起点并发执行。第一版不跨区
steal、不主动 idling、不切 route，stage window 仍由选定的 $(M_i,t_x)$ 确定。因此该
候选的离散域是

$$
(\theta,C_L,t_L,t_S),
$$

不是任意 mixed-width DAG。若有可靠的 phase service model，评分目标应为两个资源区在
同一事件时间线上的真实结束时刻

$$
C_{LS}=\max\left(C_L^{\mathrm{finish}},C_S^{\mathrm{finish}}\right)
+T_{merge},
$$

其中 dilation 必须由同时活跃的 compute、L2 refill、LLC 和 DRAM demand 推进；不能用
$\max(\sum I_i(t_L)/h_L,\sum I_i(t_S)/h_S)$ 代替。后者只适合区内 LPT 排序。

在 `AmazonC5192Cores` NUMA0 `0-95` 上，以
`dsv4-real-2048-seq70`（TP4，$H=4096,F=512,E=256$，2048 token，TopK=6，
223 active experts）验证。$\theta=48$ 时，大区含 28 个 expert/8875 routes，小区含
195 个 expert/3413 routes。最优稳健配置为

$$
C_L=C_S=48,\qquad t_L=8,\qquad t_S=1,
$$

即 6 条大-M lane 与 48 条短-route weight stream。51 轮交错复测中 strict `12x8T`
为 13.367 ms，双区为 11.625 ms，aggregate throughput 从 11.567 提升到
13.301 TFLOP/s（$+14.99\%$）；独立 31 轮复测为 13.345/11.551 ms
（$+15.53\%$）。$t_L=16$ 与 8T 相差 0.21%，低于噪声，保留 8T 作为更窄且并发 lane
更多的代表点。$\theta\in[28,192]$ 在本 histogram 上近似平台，而 $\theta=12$ 回退
16.71%，因为它把 118 个 $M=13\ldots28$ expert 错分到只有 6 条大区 lane。

低开销 trace 显示大/小区实际 compute 结束于 11.087/11.301 ms，skew 仅 0.214 ms。
相对 strict，W13/W2 core-time 从 744.95/361.04 降到 630.55/295.06 core-ms，
gather/pack 保持 18.85/18.95 core-ms。300-call PMU 长循环中，总 instructions 仅降
2.32%，但 cycles、L2 refill、memory-stall cycles 分别下降 13.92%、6.61%、29.07%；
`mem_access/cycle` 与 `l2d_cache_refill/cycle` 分别提高 19.17% 和 9.05%。这支持的机制是：
短 expert 以更多独立 1T cold-B stream 提高 memory-level parallelism，大 expert 保留
8T compute/cache reuse，两类需求重叠后减少 backend 等待；不是简单增加总内存流量。

当前模型尚未闭合该候选。isolated LPT 预测大/小区为 8.812/3.760 ms，实际区间为
10.817/11.032 ms；现有全局 working-set simulator 预测 14.932 ms，相对实测
11.593 ms 高估 28.81%。因此该实现只保留 benchmark/timeline 入口，不进入 production
planner 或 cache identity。采用前必须以 phase-local offered demand 和机器 service
capacity 重建 cross-class dilation，在其他 DSV4 routing seeds、catalog 混合分布和至少
另一台 ARM 机器上达到 absolute error $\le3\%$、selected regret $\le2\%$、无 case
回退超过 2% 的 gate。完整命令与原始计数见
`optimizations/fused_moe_sve/results/amazon_192c_dsv4_large_small_partition_20260811.md`。

##### 9.37.1 有界 M<=12 常驻流实验

双区 LPT 保留 isolated cost 降序，因此小区内的极短 expert 仍集中在时间线后半段。为验证
“让少量 cold-weight stream 从调用开始持续运行”这一假设，将小区进一步静态分为 medium
与 short 两区。给定 $0<\theta_S<\theta_L$：

$$
\mathcal J_L=\{i:M_i>\theta_L\},\quad
\mathcal J_M=\{i:\theta_S<M_i\le\theta_L\},\quad
\mathcal J_C=\{i:0<M_i\le\theta_S\}.
$$

核心满足 $C_L+C_M+C_C=T$；large 使用 $t_L$，medium/short 固定为 1T，三组独立 LPT
依赖链均从调用起点可运行。该 benchmark-only 域为

$$
(\theta_L,\theta_S,C_L,t_L,C_C),
$$

并且没有跨区 steal。若忽略 cross-class dilation，使 short 区覆盖目标 makespan
$C^\star$ 的 isolated 下界为

$$
C_C^{iso}=\left\lceil
\frac{\sum_{i\in\mathcal J_C}I_i(1)}{C^\star}
\right\rceil.
$$

DSV4 上取 $(\theta_L,\theta_S,C_L,t_L)=(48,12,48,8)$。short 集合为 71 个 M1、
3 个 M6、1 个 M10，总 isolated service 为 27.504 core-ms；代入约 11 ms 得
$C_C^{iso}=3$。实测反证该估计：4/6/8/12/16 条 short stream 的 31-round median 为
`14.295/13.001/12.158/11.971/11.851 ms`。51-round 细扫中，原双区为 11.595 ms，
12/13/14/15/16 条为 `11.891/11.900/11.850/11.881/11.789 ms`；最佳静态常驻流仍回退
1.67%。

trace 确认实验确实完成了时间整形：short 首次启动从 5.177 提前到 0.277 ms，峰值并发从
26 限制为 16。但 short W13+W2 core-time 从 133.55 增至 136.49 core-ms，medium W13
从 212.06 增至 223.92 core-ms（+5.6%）；short/large/medium 分别在
9.535/11.104/11.588 ms 结束。故 compulsory bytes 与 isolated service 不能单独确定
cold-stream 配额：在该完整静态分配中，提前 short 没有降低其自身 core-time，同时
medium W13 变慢，但这两个现象不能直接解释为单一的 cross-class 因果关系。该三静态区候选
只保留为反例，不进入 production 搜索。后续若测试
short 完成后向同宽 medium pending suffix 的非阻塞释放，必须相对原双区重新通过 2% gate。

dependency-only 控制保持 16-stream 的 expert membership、核心、宽度和 window 不变，只让
每个 short 根任务依赖所有 medium lane 终点。相对 eager，medium W13/W2 从
223.92/109.01 降至 148.88/80.23 core-ms
（-33.5%/-26.4%），large W13 从 322.08 降至 299.24 core-ms（-7.1%），short
W13+W2 从 136.49 降至 57.30 core-ms（-58.0%）；medium 完成由 11.588 提前到
8.237 ms。delayed wall time 仍从 eager 11.784 增至 12.340 ms，因为去掉 overlap 后 short
形成串行尾部。该实验只能证明 task service time 随 active mix 显著变化；它同时改变了活跃
核心数、wave 结构和串行尾部，因此不能把完整回退因果分解为“interference 主、idle 次”。

随后用严格配对控制固定任务、核心和 lane 宽度：同一批 48 个 M28、48 个 M1 expert 在
48 个 1T lane 上，每核恰好执行一个 M 和一个 S，只比较全部 `M->S`、全部 `S->M`、半区
反序和奇偶核反序。核心 48--95 上的 41-round median 分别为
`7.1902/7.1724/7.2368/7.2291 ms`。为避免大 expert 成为 full-DAG critical path 而遮蔽
前景差异，进一步把同一组 DSV4 long chain 放到独立进程，只占用 0--15、0--31 或 0--47
核心并持续执行，计时仍只覆盖 48--95 核的 M/S 前景。全 `M->S` 前景随背景核数
0/16/32/48 由 `7.1902` 增至 `8.6269/9.8333/10.6404 ms`（+20.0%/+36.8%/+48.0%），
证明背景争用真实存在；对应奇偶交错为 `7.2291/8.6770/9.7385/10.6871 ms`，没有随压力
增加的单调收益。只有 32 核背景点约快 1%，两次追加 paired median 为 +0.31%/+1.39%，
但 P10--P90 均约跨零 $\pm6$ 个百分点；16 核为负、48 核中性。因此没有证据说明
`M+M -> S+S` 与 `M+S -> S+M` 本身存在稳定显著优劣。争用是调度决定 active mix 后产生
的代价函数，不能作为独立于调度的原因；均匀 M/S 混合不进入 planner 约束，只可在 core
allocation 与 critical-path balance 等价时作 tie-break。当前只确认完整三静态区方案回退和
service rate 状态相关；后续必须分别固定 allocation、release 与 tail 结构再做因果分解。

据此固定 production 决策：不要求 medium（$13\le M\le48$）与 short（$M\le12$）在时间上
均匀混合；任务顺序只服从预测 critical path 与 lane-load balance。均匀混合最多作为零代价
确定性 tie-break，不得为此静态预留 core region，也不得覆盖实测或模型选择出的更快分组顺序。

#### 9.38 固定 lane membership 的时间交错顺序

原 strict planner 先按 route 降序做 isolated-time LPT，并把每条 lane 的 expert 按同一
降序执行。异构 workload 因此容易形成“先同时运行大 expert、后同时运行小 expert”的
时间聚集。第一版不改变 LPT 的 lane membership、线程宽度、连续 core interval、stage
window 或 Plan V2 DAG 形状，只给每条已有链增加一个二值方向：

$$
o_l\in\{+1,-1\},\qquad
\pi_l(o_l)=
\begin{cases}
\pi_l^{LPT}, & o_l=+1,\\
\operatorname{reverse}(\pi_l^{LPT}), & o_l=-1.
\end{cases}
$$

其中 LPT 的插入顺序为 route 非增，因此反序 lane 会提前小 expert、延后大 expert。在线
候选固定为三个 seed：

$$
\mathcal O=\{(+1,\ldots,+1),\;o_l=(-1)^{l+1},\;o_l=(-1)^l\}.
$$

三个 DAG 都由现有完整 event-time contention model 评分，选择

$$
o^*=\arg\min_{o\in\mathcal O}\widehat C(o).
$$

只有当候选满足

$$
\widehat C(o)<\widehat C(o_{best})-
\max(10^{-6}\ \mathrm{ns},10^{-10}|\widehat C(o_{best})|)
$$

时才替换当前方案；因此浮点平局确定性保留原 LPT。uniform full-call anchor 不包含顺序
语义，命中时不做反序搜索。Python reference planner 与 native C++ cold planner 使用同一
seed 顺序和阈值。选中的 `assignment_order` 是 planner/cache 内部诊断元数据，不进入
Plan V2 bridge；`PlannedMoE` cache hit 重新计算便宜的 LPT membership 后直接应用缓存的
方向，不重复运行 event-time DAG 评分。

这个候选仍是 non-idling fixed-lane DAG：每条 lane 的前一个 expert 完成后立即启动下一
个；没有跨 lane 迁移、route 切分、抢占、动态扩缩或新的 runtime 字段。它也不是任意
permutation 的局部最优证明。曾评估在最好 seed 后逐 lane 做一次坐标下降，但当前 192C
profile 上 DSV4 的预测改善只从 10.01% 增至 10.03%，tiered hotspot 从 8.52% 增至
8.72%，却显著增加 cold-plan DAG 评分次数，故第一版不保留该搜索。

当前 TP4/F512 96-core empirical profile 的 strict-only 离线评分如下；数值是同一模型对
原 LPT 与三 seed 最优值的比较，不是硬件实测加速：

| workload | 原 LPT 最优 | 时间 seed 最优 | 预测改善 |
| --- | ---: | ---: | ---: |
| captured DSV4 | 9.784 ms | 8.894 ms | 10.01% |
| long/short bimodal | 9.687 ms | 9.461 ms | 2.39% |
| tiered hotspot | 8.258 ms | 7.610 ms | 8.52% |
| active-set 128 | 10.076 ms | 10.076 ms | 0.00% |

合成 phase-contention 单测把同类 phase 同时运行设为 2x slowdown；原始两条
`large->small` lane 的 makespan 为 24，奇偶反序后为 20，同时逐 lane expert 集合保持
不变。无 contention 的平局单测则逐 task 保留 LPT。现有 table/formula、strict/auto/
forced-tail、bounded-tail 的 Python/native parity 全部保持。同一开发机上的 planner-only
smoke 使用上述 192C profile 时，native 8-worker cold median 为 bimodal 1.057 ms、DSV4
3.588 ms；缓存方向后 warm median 为 1.121/2.209 ms，且单测确认 cache hit 不调用
`dag_makespan`。这些数字只度量 planner 开销，不是目标 ARM kernel 性能。

边界必须与 9.37 一起解释：当前 cross-class service 的绝对时间 gate 尚未闭合，所以
上表只能证明候选在现有模型目标下可取，不能证明真实 E2E 同比例改善。

2026-08-11 在 `AmazonC5192Cores` NUMA0 `0-95` 上完成第一轮真实 A/B。构建使用当前
未提交开发树（base `f6debf5d1b8d`）、Linux AArch64 SVE JIT exact-M、有效 `-O2`、
32 MiB HugeTLB packed weights、TP4 `H=4096,F=512,E=256`、BF16 route store；每个
case 做 7 次 warmup 和 51 对随机交错调用。基线与候选的 shape、core interval、线程宽度、
stage window、tail policy、输入和权重完全相同，只把候选的 `assignment_order` 恢复为
`lpt`；输出逐位一致。下表以逐对 speedup 的中位数为主，括号内为 P10/P90：

| execution | workload | order | LPT median | temporal median | paired speedup |
| --- | --- | --- | ---: | ---: | ---: |
| strict | tiered hotspot | reverse odd | 8.005 ms | 8.033 ms | -0.35% (-2.27%, +2.04%) |
| strict | long/short bimodal | reverse even | 11.954 ms | 12.769 ms | -5.90% (-9.14%, -0.56%) |
| strict | captured DSV4 | reverse even | 13.274 ms | 12.248 ms | +8.34% (+5.06%, +9.55%) |
| tail pool | long/short bimodal | reverse even | 9.928 ms | 9.916 ms | +0.11% (-0.71%, +0.90%) |
| tail pool | captured DSV4 | reverse even | 13.255 ms | 11.983 ms | +10.65% (+9.01%, +11.41%) |

DSV4 使用另一组 seed 的独立 51-pair 复测为 `13.277 -> 12.004 ms`，paired median
`+10.64%`，51 对中 50 对获胜。uniform 的两个 bridge 相同，paired median 为
`-0.01%`，作为本轮测量噪声对照。active-set workload 因 route 相同仍逐位选择 LPT，
不会产生 runtime 计划变化。

该结果同时验证收益与否定无条件启用：模型正确选中 DSV4，却把 tiered 的中性结果预测为
`+8.52%`，并把 strict bimodal 的首轮 `-5.90%` 回退预测为 `+2.39%`。因此第一版未通过
held-out regression `<=2%` gate，不能作为无条件 production strict 顺序。

随后对 strict bimodal 做两组进程内 phase trace。反序的 GEMM 主体并未回退：两组中
compute end 都从 LPT 的 `11.734/11.836 ms` 提前到 `10.933/10.785 ms`，W13/W2 task
median 均下降，compute core-ms 也从 `910.8/903.9` 降为 `891.3/886.1`。差异全部出现在
ready-token 尾部：

| process state | order | compute end | compute 后 late tokens | ready-merge tail |
| --- | --- | ---: | ---: | ---: |
| fast | LPT | 11.734 ms | 47 | 0.037 ms |
| fast | reverse even | 10.933 ms | 2040 | 0.347 ms |
| slow | LPT | 11.836 ms | 59 | 0.051 ms |
| slow | reverse even | 10.785 ms | 2041 | 1.939 ms |

该分布的 5 个长 expert 都有 `M=2040`，TopK=6 的多数 token 需要这些长 expert 和一个
短 expert。LPT 先完成长 expert，再由 174 个 `M=12` expert 逐批发布 token；时间反序则
提前完成短 expert，并在最后几个长 expert 完成时集中使约 2040 个 token ready。runtime
在最后一个 route slice 的 `local_tid==0` 上串行执行 `publish_ready_tokens(expert_routes)`：
对每个 route 扫描 TopK completion，并写 `token_ready`；同时 96 个 fixed owner 轮询各自
token range。该 publication/owner-drain 相位不在当前 expert phase cost model 内，且在
实机上出现 `0.35--1.94 ms` 双稳态，足以覆盖反序约 1 ms 的 compute 收益。

因果对照关闭 ready-token early merge、保留同一 strict plan 和最终完整 merge。两个独立
51-pair 进程分别从 `12.066 -> 11.386 ms`、`12.023 -> 11.363 ms`，paired median 为
`+5.88%/+6.07%`，P10 仍为 `+3.83%/+4.56%`。因此 bimodal 回退不是 GEMM/cache 时间
交错本身，而是 temporal order 与 fixed-owner ready-token publication/drain 的组合效应。
	进入 production 前应让 planner 同时建模 merge-readiness，或在会形成大 ready burst 的
	temporal plan 上关闭 early merge；只按 expert GEMM DAG 评分仍不满足 2% gate。

第一版采用 2.5 节的 routing-shape/route-bound 保守 gate，而不引入完整 merge queue：
选定 strict compute plan 后，对高覆盖 expert 用 route count 减去全部更晚 expert route
occurrence，得到其 ready burst 下界；若该下界之外的 token 至多为默认一轮 owner drain
（$2T$），直接选择连续 final merge，否则保留 auto。
该 gate 不改变 temporal candidate 分数、shape cache 或 Plan V2 ABI，也不对 early merge
作正向收益判断。当前 selected strict `6x16T` bimodal plan 的 burst 下界为
$L^*=2016,N=2048,T=96$，故 $N-L^*=32<192$，plan-only 已选择
`early_merge=false`；同一检查对 DSV4/tiered 的下界只有 `917/384`，保持 auto。
AmazonC5192Cores NUMA0 随后以 32 MiB HugeTLB、7 次 warmup 和 51 次成对交错调用
复验完整 9-case catalog。三个改变 lane 顺序的 strict case 中，bimodal、DSV4 和
tiered hotspot 的 median 分别为 `+6.28%/+8.80%/-0.56%`；bimodal paired median/P10
为 `+6.39%/+3.04%`。六个保持 LPT 的对照 median 绝对偏差均不超过 `0.89%`。
最终只有 bimodal 的 selected plan 从 auto 改为 `early_merge=false`，其 LPT 对照仍为
auto；DSV4/tiered 两个顺序均保持 auto。由此关闭 192C host 的 2% held-out gate，
但不替代跨机器验证。

#### 9.39 DSV4 尾部四类候选闭合

2026-08-12 在 `AmazonC5192Cores` NUMA0 `0-95` 上验证 2.3.3。输入为 captured
DSV4 TP4/F512（2048 token、TopK=6、256 experts、223 active），基线保持 9.37 的
`48C x 8T` large 与 `48C x 1T` small 双区；同一对比中的权重、window、early
merge、HugeTLB 和输出 allocation 不变。结果如下：

| 候选 | 实测结果 | 决策 |
| --- | ---: | --- |
| 完整 `1/2/4/8T` cohort 重组 | `-0.849%` | 删除 runtime/schema/API 实现 |
| 非阻塞 `2/4T` 重组 | `-0.613%` | 没有自然形成 2T/4T task，删除 |
| 1T LPT 改为 M 升序 | LPT 相对 `-0.069%` | 噪声内，保持现有 LPT |
| cost-model 自动 suffix DAG（E219/core40） | `-0.077%` | 模型误选，删除 planner 候选 |
| 实际完成时触发的 bounded residual-M | `-0.358%` | 在线检测/交接抹平收益，删除 runtime 协议 |
| 静态 residual-M（E218/core40） | `+0.48%--+1.20%` | 保留 Lab comparator；未过 2% gate |

完整 cohort 重组的 51-pair 绝对中位数为 `11.497479 -> 11.595985 ms`；75 个
pooled expert 中 73 个仍为 1T，只有 2 个变为 4T。独立 trace 的 tail idle 从
`70.94` 降为 `44.50 core-ms`，但 internal idle 从 `24.60` 增为
`34.89 core-ms`，说明减少尾部面积不等价于降低 makespan。非阻塞窄重组进一步证明，
work-conserving 的 1T worker 会在 cohort 形成前消耗掉绝大多数机会。

静态 residual-M 对同一 E218/core40 组合做五次独立交错复测，收益为
`1.093%/0.482%/1.205%/0.685%/1.193%`，均值约 `0.93%`；E217、E208、E216、E219
分别为 `-0.147%/-0.190%/+0.046%/-0.153%`，说明 target/donor 选择不可忽略。
在线版本即使把等待限制为 300 us，11-pair 仍从 `11.742341` 回退到
`11.784479 ms`。因此主路径保持不变，P0 不关闭；只保留显式静态比较器和结果文档，
不保留任何失败候选的 production/Lab runtime 分支。完整方法与命令见
`optimizations/fused_moe_sve/results/amazon_192c_dsv4_tail_candidate_closure_20260812.md`。

#### 9.40 Arm-codex 多 LLC machine schema-v2 回放

2026-08-13 在 `Arm-codex-internal` NUMA3 `240-319` 上确认 sysfs 拓扑：
`240-279` 与 `280-319` 为两个独立 70 MiB LLC 域，私有 L1D/L2 为
64 KiB/1.25 MiB。旧 80C probe 只从首核读取 LLC，因此错误记录 rank 容量为
70 MiB；schema-v2 回放按域求和修正为 140 MiB，并在 provenance 标记该修正。

40C 单域 LLC B-only 满载 service 为 `602.160 GB/s`。假设两个等构域并行，域内
服务和为 `1.204321 TB/s`；80C rank 实测为 `1.164655 TB/s`，即 rank/shared-fabric
保留 `96.706%`，组合误差 `3.294%`。这支持
$\min(\sum_dB_d,B_{LLC,rank}^{sat})$ 的首版结构。相比之下，旧单一 smooth LLC
curve 在相同采样点上 MAPE/max 为 `41.8%/87.2%`。

DRAM 不具备这种可加性：独立 40C probe 为 `305.857 GB/s`，80C 为
`379.940 GB/s`，只增加 `24.22%`，故继续作为 NUMA-rank shared service。LLC 与
DRAM 的 piecewise 曲线在采样点内残差为零只是插值恒等式；尚未采集的 balanced
`x+x` 放置与 held-out width 仍是精度 gate。对 80C 已有内部宽度做
leave-one-sampled-width-out（非独立复测）的采样密度诊断，LLC MAPE/max 为
`8.70%/17.80%`，DRAM 为 `7.71%/21.60%`；DRAM 最大缺口在 40T，说明下一轮仍需
补 balanced placement/中间宽度，而不能只保留端点。当前 planner DAG 不携带 CPU interval，
因此这次只关闭 schema、拓扑服务 API 与旧数据回放，不关闭 production absolute-time
或 placement-aware scheduling gate。

## 10. 同步规则




发生以下任一变化时，必须同步更新本文档：

- job 粒度从 expert 改为 phase，或反向合并；
- 线程宽度定义、最大线程数或可行宽度集合变化；
- 是否允许抢占、迁移、动态调整线程数或主动 idling；
- CPU 核心硬约束、affinity 或其他真正可执行性约束变化；
- $I_i(t)$、$D_i(\mathcal Z)$ 的定义或 active configuration 状态发生变化；
- rank-local 与跨 rank 资源耦合方式变化；
- 目标函数加入或移除 planning、dispatch、communication、scatter、combine；
- kernel variant 从全局选择改为 per-expert 选择；
- full-stage/N-owner 几何、backend N tile 或 team-width 到 owner stripe 的映射变化；
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
| 2026-07-14 | v0.3 | 将资源需求记号明确为 $d_{i,r}(t)$；补充下标含义、量纲一致性及总流量与瞬时 cumulative demand 的区别。 |
| 2026-07-14 | v0.4 | 明确 $\mathcal E$ 表示输入中的强制无环数据依赖，而不是 planner 选择的 lane 执行顺序；完整 expert 粒度下取空集。 |
| 2026-07-14 | v0.5 | 明确 $a_i(\tau)$ 是由执行区间推导的资源占用指示函数，并说明非抢占连续区间语义及 CP-SAT 等价编码。 |
| 2026-07-14 | v0.6 | 解释 cumulative resource 是同时活跃需求之和与可释放的 renewable capacity；区分 LLC 硬容量和带宽软争用。 |
| 2026-07-14 | v0.7 | 将问题准确描述为混合离散-连续非凸优化；给出非凸反例、NP membership 验证和从 `3-PARTITION` 到判定版本的强 NP-completeness 归约。 |
| 2026-07-14 | v0.8 | 严格区分变量与其可行域，并用整数线程宽度的凸组合反例说明离散域非凸。 |
| 2026-07-14 | v0.9 | 增加可解释 `T_iso` roofline shadow model：按 M12/tail panel 定义 W13/W2 FLOP 与 N-split 流量，并明确单一延迟样本无法同时识别算力和 L3 带宽 ceiling。 |
| 2026-07-14 | v0.10 | 将 GEMM shadow 扩展为 microkernel-aware ECM：区分逻辑/计算/packed/store 行，按 asm 统计 BFMMLA 与 A/B loads，分离 L1/private/shared-cache 和 epilogue，并记录 V3 长 route 留出验证。 |
| 2026-07-14 | v0.11 | 在 192-core Neoverse-V3 NUMA0 上生成 TP4/TP2/EP4/EP2 split/no-split 的 96T schema-v2 校准；保持 active $T_{\mathrm{iso}}$ 公式和 production 线程剪枝不变，并记录 64-expert route-2040 full-call cliff。 |
| 2026-07-15 | v0.12 | 增加 split-W13 owner-cache 工作集 shadow：由独立 stream 饱和度给出并发下界，由 N-split 每核 L2 way budget 给出容量上界，再用 $T_{\mathrm{iso}}$ headroom 选择最小稳健工作集；记录 V3 PMU、route 1020/2040 留出结果及短 route 失效边界。 |
| 2026-07-15 | v0.13 | 将 GEMM shadow 重构为实现弱相关三层模型：算法层定义 useful work/compulsory traffic，kernel mapper 产生统一 physical demand，目标机器实测 profile 提供 service response；SVE M12 细节移出通用核心，active planner 与剪枝不变。 |
| 2026-07-15 | v0.14 | 将原始问题改为 256-expert route-task 实例对应的问题族：CPU 核心数是基础硬约束，真实 isolated time $I_i(t)$ 与动态 contention slowdown $D_i(\mathcal Z)$ 定义执行速率；LLC/带宽从硬 cumulative constraint 降为 slowdown/candidate-pruning 因素，并同步复杂度、solver 边界、剪枝表和验证要求。 |
| 2026-07-15 | v0.15 | 增加最优解存在性命题：由串行调度得到有限 horizon，将固定线程宽度与事件顺序的子问题表示为紧线性可行域，再由 Heine--Borel 和 Weierstrass 极值定理证明全局最小 makespan 必然取得；同时明确 $I,D$ 必须在所有候选配置上有限且为正。 |
| 2026-07-16 | v0.16 | 增加显式 packed-B byte-window kernel variant：按 SVE N tile 推导 W13/W2 range 数、实际最大窗口和有效线程上界；明确 range 间无 barrier、owner-scatter ownership、M12 短 route 边界，以及该 variant 在重新校准前不进入 production planner。 |
| 2026-07-16 | v0.17 | 增加可选 ready-token combine 外层：定义 TopK release time、单线程 merge CPU 约束和含 combine 的 makespan；记录 expert-first async heuristic、25% team-load 生效门槛、planner/cost-model 边界及 192-core NUMA0 首轮验证。 |
| 2026-07-16 | v0.18 | 将通过验证的 async ready-token executor 设为 SVE FP32 direct-route 默认；保留 1.25 team-load 门槛和环境变量值 0 的 post-expert fallback，不改变 planner 决策空间或现有 cost tables。 |
| 2026-07-16 | v0.19 | contention/working-set profiler 改为复用预分配 native `out`；将 allocation 与 first-touch 排除出 $I_i(t)$ 和 $D_i(\mathcal Z)$，并标记旧大 route allocate-per-call profile 不可用于 GEMM contention 校准。 |
| 2026-07-16 | v0.20 | 在 192 核双 NUMA 上以两个 96-core 同步 rank 生成 TP4/F512 split-W13 稳态 profile；记录 route-dependent 最优 lane shape、35.89 TFLOP/s 长 route 聚合吞吐和跨 NUMA pairwise-max 必要性。 |
| 2026-07-17 | v0.21 | 将双 NUMA TP4/F512 split-W13 校准扩展到真实 256 local experts，补齐 `96x1T` 等形状；记录 M=12 的 `24x4T`/`48x2T`/`96x1T` 宽平台，并禁止将 E64 full-call 排序外推到 256-expert TP4。 |
| 2026-07-17 | v0.22 | 增加 NUMA0 M12 单满波冷 packed-B 带宽校准：256 份权重窗口轮换、1--96 个 1T experts、峰值 352.1 GB/s；记录 50%/75%/90%/95% 稳健带宽所需的 12/24/64/86 线程阈值，不改变 active planner 或剪枝。 |
| 2026-07-20 | v0.23 | production SVE compute 改为 Xbyak exact-M1--M12：定义 $m_c=2\lceil M/2\rceil$ tail mapper，profile identity 增加 implementation/tail policy，route grid 补齐 1--12；记录 V1/V3 bit-exact 与稳态性能验证；planner `auto` 只原子选择完整 variant pair，禁止 JIT 和 static bucket profile 混用。 |
| 2026-07-22 | v0.24 | 增加论文评测用 uniform、active-set sweep、tiered hotspot 和 long-short bimodal 全局 TopK workload；补充合法 histogram 约束和验证覆盖说明，不改变 planner 可行域、cost model 或 production 剪枝。 |
| 2026-07-23 | v0.25 | 增加 V3 单核 packed-B service ceiling 40.04 GB/s；区分 exact-M lane、useful-compute、physical-issue 和 memory efficiency，并记录冷权重 M1--M12 验证；不改变 active planner、contention table 或剪枝。 |
| 2026-07-26 | v0.26 | 将全局 packed-B byte-window 接入 schema-v2 与 production planner：variant identity 包含目标字节和 W13/W2 实际 range 数，stage model 分别推进两段 range，联合搜索 `(window, core shape)` 并透传 runtime option；旧 profile 映射为 window=0，未提供新实测表时决策不变。 |
| 2026-07-26 | v0.27 | 将 x86 BF16 auto backend 改为优先 AMX 并回退 AVX-512；增加按 expert route 数选择 m2n2/m1n4 的确定性 M=76 policy，以及按 K-padded byte budget 自动推导 1 MiB/512 KiB cache window；同步 per-expert variant 公式、剪枝表、C8i 正确性/性能验证和 policy 可迁移性边界。 |
| 2026-07-26 | v0.28 | 将 SVE 的 team-N ownership 引入同步 x86 AVX-512/AMX executor：active expert 不足时按 route/width 贪心组 team，2x 且至少 64-row 的强偏斜 route 使用有序 waves，均衡 route 保留 expert queue；定义精确 N-range、team-width 与 wave 公式，记录 8-core C8i 1/2/4/8T 验证，并明确该 mapper 尚不进入 planner candidate space。 |
| 2026-07-26 | v0.29 | 引入向后兼容的 async Plan V2：增加 strict execution mode、离散 allowed-width CSR、preferred/min/max、NUMA/stage/range/resize 字段及强校验；production planner 仍生成 singleton width 并严格降级到现有 fixed-interval native DAG，因此 moldable 语义、公式、剪枝和 cost model 保持不变。 |
| 2026-07-26 | v0.30 | Plan V2 接入 ARM native strict entrypoint，并增加显式 whole-expert tail_pool placement：对齐线程组只在覆盖 fixed tasks 全部完成后重组，运行中宽度仍固定；planner 默认 strict，tail pool 仅作为不参与 cost-model 排序的强制实验 bridge。同步 placement/依赖/对齐约束、剪枝表和验证要求，并记录 192-core 主机 NUMA0 的 bit-exact 与 long-short bimodal 性能验证。 |
| 2026-07-26 | v0.31 | 用当前 ARM source/extension hash 原子刷新 192-core 双 NUMA TP4/F512 JIT exact-M split/no-split 配对表；记录绝对时间漂移、96T 不稳定性、公式拟合误差、旧选择 regret 和默认 workload planner 变化；公式形式、candidate space 与 production 剪枝不变。 |
| 2026-07-26 | v0.32 | 增加 async benchmark-only W13/W2 独立 packed-B 窗口覆盖；在 V3 双 NUMA 上验证另一机器的 1/0.5 MiB-per-worker 规则只对中等 route 有利、长 route 回退，并用二维扫描确认 stage 最优值同时依赖 route/team width；公开 API、profile identity、公式、planner 候选和剪枝均不变。 |
| 2026-07-26 | v0.33 | 增加 planner-compatible 解析 SVE MoE backend：由 exact kernel demand、L2/LLC 容量模型、分层 service curves 和共享资源 event simulator 计算 isolated/contention 时间；machine calibration 不包含 route/thread 或 shape 表。analytic backend 生成 homogeneous/双宽度 shape 并保持 opt-in，旧 empirical backend 继续作为 production 默认和 holdout oracle。 |
| 2026-07-26 | v0.34 | 将 whole-expert dynamic tail-pool 纳入 production planner 默认候选：在 workload 内搜索短-route threshold、`1/2/4T` 自动 pool width 和 strict 竞争 head shape，用 fixed-group release + LPT list scheduling 构造 surrogate DAG，再由现有 contention model 评分；runtime 保留在线领取，新增 strict opt-out、全宽度 forced override、threshold-eligibility 隔离缓存和 strict/auto benchmark 对比。 |
| 2026-07-26 | v0.35 | 将 schema-v2 empirical cold planner 等价迁移到 C++，并以 OpenMP 在 strict 和 dynamic 两阶段分别并行评分候选；固定 candidate index 保证跨线程数确定性，GIL 在搜索期间释放，Python/analytic fallback 与 cache-hit 路径保留。192-core 主机前 96 核上，默认 8T 将 bimodal/captured cold search 从 Python 59.784/274.342 ms 降至 1.421/3.488 ms；该变更不修改公式、候选空间、剪枝或 runtime 调度语义。 |
| 2026-07-26 | v0.36 | Plan V2 增加 per-task W13/W2 packed-B byte-window override；planner 可在选定 task/DAG 后应用命名的确定性 $g(M,t)$，不扩大或重新评分搜索空间。记录 AmazonC5192Cores NUMA0 TP4/F512 静态策略及 7 个 E2E workload：active-set-128/64/32 分别提升 28.14%/10.38%/1.66%，未覆盖 case 最大观测回退 0.32%；该机器专用策略保持 opt-in，待独立窗口 calibration 后才可进入 scored production policy。 |
| 2026-07-26 | v0.37 | 将 `amazon_c5_192c_tp4_f512_v1` 设为精确 profile-bound 默认：只匹配双 NUMA AmazonC5192Cores TP4/F512、96-core rank、SVE JIT exact-M split-W13 identity，并按 selected model 分别解析，no-split 和其他 profile 不受影响；增加 `use_default_stage_window_policy=False` opt-out。NUMA1 active-set-64/128 独立复测分别提升 10.58%/29.71%，与 NUMA0 一致；该 runtime policy 仍不进入 cost-model 评分或跨机器外推。 |
| 2026-07-26 | v0.38 | 将确定性 $g(M,t)$ 前移到候选执行建模：empirical backend 按实际 W13/W2 range 和瞬时 working set 进行 contention event simulation，analytic backend 同时重算 range/cache/DRAM demand 与 isolated time；命中 override 的 workload 禁用 global-policy full-call anchor，Python/native cold planner 共用同一组已解析 geometry。shape、tail-pool 和 kernel variant 候选均未增加，window 仍不是自由搜索变量。 |
| 2026-07-26 | v0.39 | 实现离线 isolated CP-SAT oracle v1：对每个 expert 选择一个非抢占固定线程宽度，以 `T_iso` optional interval 和 CPU cumulative constraint 求最小 makespan；增加 mode dominance、整数时间量化、可行 incumbent/best-bound regret 区间、strict planner 同口径评估和独立 CLI。该工具为可选 OR-Tools 依赖，不改变 production planner、runtime 或 contention 模型。 |
| 2026-07-26 | v0.40 | 在 AmazonC5192Cores 的 TP4/F512 96-core rank 上用 9 个默认 workload 验证 isolated CP-SAT oracle，并与双 NUMA 当前 runtime 配对最大值比较。active-set-8 与 long/short bimodal 分别证明 `7.9300/6.1316 ms` 最优；当前 bimodal tail-pool 的 isolated regret 为 0%，但 wall time 仍高 78.3%，确认 v1 只能界定 isolated scheduling gap，不能充当无资源容量约束的硬件性能证书。 |
| 2026-07-27 | v0.41 | TP/EP evaluator 增加跨 rank lifetime phase 转换：多 rank 活跃时使用 concurrent-rank profile，倒数第二个 rank 完成后按当前 phase 剩余比例切换至 matching single-rank companion；不重放 setup/dependency，不扩大 planner 候选或改变剪枝，缺少 companion 时保守回退旧上界。 |
| 2026-07-27 | v0.42 | 用当前 JIT exact-M kernel 生成 32-core/rank EP2 H4096/F2048/E32 single/dual 配对表，并在 tiered-hotspot 上验证 lifetime 转换将 55.605 ms 修正为 54.770 ms（-1.50%）；同步刷新 8-core standalone 与 192-core TP4/F512 single/dual empirical profiles。 |
| 2026-07-29 | v0.43 | 增加 benchmark-only 全局两阶段 Plan V2：W13/W2 可独立搜索 strict/tail-pool shape 和确定性 stage window，以显式全局 barrier 连接；增加 matched 对照和首版 `setup+2/3 compute` 与 `1/3 compute` stage surrogate。192-core NUMA0 的 9-case 验证中 independent 相对 matched 在三个异构 stage-plan case 提升 3.28%--7.52%，但仍在 9/9 case 落后 whole-expert 2.29%--6.26%，因此保持实验入口且默认 runtime 不变。 |
| 2026-07-29 | v0.44 | 增加实验性 W13/W2 单边界 elastic Plan V2：W13 保持 planner selected fixed team，W2 可非阻塞或在有限 timeout 内取得同 NUMA aligned preferred cohort；失败保留原 team，不抢占、不复制 intermediate。增加显式 `8T->16T`/`2T->8T` planner bridge、自然机会/等待/回退统计和 benchmark 入口；production 搜索空间与默认 strict/tail-pool 不变。 |
| 2026-07-29 | v0.45 | 为 `timeout=0` 增加无中央锁的原子 ownership 快路径，并在 192-core NUMA0 完成 `2x8T->16T`、`4x2T->8T` 验证。自然机会率仅 `0--6.53%`，有限等待虽提高 preferred assignment 仍无 E2E 收益，且 boundary handoff 在多短任务上显著；因此 elastic 继续保持显式实验模式，不进入 cost-model 或 production 候选。 |
| 2026-07-29 | v0.46 | 为实验 elastic Plan V2 增加显式 `task_preferred_core_begins`：W2 target 可包含 source 或与其完全不相交，必须按 preferred width 对齐且保持同 NUMA；runtime 先完整取得 destination，再释放 source-only workers，并允许同一次 scheduler pass 用释放的 source 组成另一个 cohort。planner bridge/benchmark 可只选择尾部 task 并指定离散 target，strict/tail-pool 和默认 `-1` 派生行为不变。 |
| 2026-07-30 | v0.47 | production planner 增加一次性 bounded tail repartition：只改写恰好两个、无后继的第二波 whole-expert task，在 96-core 域搜索 `24/32/48T` fixed interval，并用覆盖 interval 的首波 blocker 重建 strict DAG；无合法候选或显式关闭时回退原 strict。Python/native cold planner、cache 和 ranking 同步携带 tail metadata；无 width 表时只允许长整 M12 isolated formula 插值。记录 `6x16T -> 2x{24,32,48}T` 静态验证和 51-run production E2E：auto 选择 32T，相对 strict 提升 12.22%，离实测最优 24T 为 0.93%。 |
| 2026-07-30 | v0.48 | 为 bounded tail 增加 exact-layout full-call anchor：签名绑定 uniform route、root lane 顺序/物理起点、tail 起点/宽度、kernel hash 和 stage-window policy，只做 exact-route lookup；未命中继续走 stage-aware DAG simulator。当前 extension 的 101-run 校准给出 24/32/48T 为 9.797/9.997/9.866 ms，Python/native planner 由错误的 32T 修正为 24T；校准后 51-run production auto 实测 9.842 ms，与 anchor 相差 0.46%；$I(M,t)$、普通 shape derate 和其他 workload 不变。 |
| 2026-07-30 | v0.49 | Plan V2 strict task 增加连续 route-slice 语义，允许同一 terminal expert 在 M 维拆成两个 disjoint task；native runtime 对每个 slice 独立 gather/W13/W2/direct route store，并在全部 slice 完成后才发布 expert completion。bounded-tail planner 只在 exact-layout anchor 命中时生成 `2 expert x 2 slice x 24T` grouped 候选，使 active-set-8 的尾波占满 96 核；显式 51-run 实验从 whole-expert 24T 的 9.804 ms 降至 8.710 ms，production auto 复测为 8.753 ms、相对同轮 strict throughput 提升 27.73%，未校准形状保持原候选域。 |
| 2026-07-30 | v0.50 | async ready-token executor 改为同一 resident worker job 排空 queue：expert 阶段保持单 token 领取，compute 完成后默认以两 token batch 和一 token lookahead 预取收尾，不再启动连续 final merge；保留 drain/merge/batch/prefetch 环境变量 fallback。AmazonC5192Cores NUMA0 active-set-8 的 101-run 中位数从旧路径 8.691 ms 降到 8.637 ms，P90 从 8.963 ms 降到 8.706 ms；batch sweep 未证明预取有独立显著收益，planner/cost-model 决策空间保持不变。 |
| 2026-07-30 | v0.51 | Plan V2 增加 plan-level `early_merge` 三态控制并下沉为 native `-1/0/1`；strict planner 复用 active expert DAG simulator 的逐 task 完成时刻，在聚合 expert finish 于 1 ns/1 ppm 内相同时写入 `false`，统一使用全 worker 连续 post-expert merge，其余候选保持 auto，planner 不强制 on。环境变量仍是全局 kill switch，elastic 拒绝 on；同步更新 API、模型公式、剪枝表、验证要求和 contract tests，combine service time 仍未进入评分。 |
| 2026-07-31 | v0.52 | 增加默认关闭的 strict suffix-steal runtime：保留 planner 前缀，只允许同 NUMA、同宽 fixed team 领取 peer lane 最后两个 whole-expert task，不抢占、不改宽度，并与 ready-token drain 共存。NUMA topology 增加进程内缓存；192-core NUMA0 的 uniform/active-set-128/captured route 分别提升 0.46%/2.04%/0.23%，tiered hotspot 持平。该动作暂不进入 planner candidate、cache identity 或 cost-model 评分。 |
| 2026-07-31 | v0.53 | async ready-token merge 从全局 claim queue 改为固定 owner：logical worker 沿用 final merge 的连续 token range，在每个 expert task 边界及无可运行 expert 时扫描本区间的 ready token，compute 后由同一 resident worker 排空本区间；保留 drain/batch/prefetch 和 final fallback 契约，不允许 owner 间 merge stealing，planner 三态和 cost-model 搜索空间不变。 |
| 2026-07-31 | v0.54 | 新增独立 cold-phase CP-SAT oracle：以 M12 isolated reference 将 expert 分成 cold packed-B/steady phase，master interval 在共享资源等待期间继续占用原 team，并对 cold phase 增加单 NUMA DRAM cumulative constraint；同模型比较固定 `12x8T` lane DAG 与 `1/2/4/8/16T` mixed-width 域。192-core TP4 profile 的 DSV4 fluid-aggregate 求解给出 `27.7%--49.8%` surrogate headroom；range-level 模型尚未收敛。该工具不接入 production planner。 |
| 2026-07-31 | v0.55 | 将 cold-phase incumbent 降为连续 core strict Plan V2：二维 CP-SAT 固定 oracle 时间区间并求物理 placement，per-task release gate 保留主动 idling，非零 release 仅用于 strict 实验且关闭 tail-steal。192-core NUMA0 的 DSV4 51-run 验证中，fixed/mixed 实测为 `14.854/20.869 ms`，mixed 相对 fixed throughput 回退 `28.82%`，而 `0.25x` release 相对 eager 仅差 `0.03%`；因此首版 cold-only surrogate 被 runtime 反证，不进入 production planner。 |
| 2026-08-01 | v0.56 | 用 stage trace、width/concurrency sweep 和多核 PMU 定位 cold-phase mixed runtime 回退：mixed 实际平均 active cores 仅 `57.18/96`，W13/W2 峰值 active window 达 `201.5/161.5 MiB`，L2 refill/LL miss 相对 fixed 增加 `19.1%/21.5%`；`min-width=4` 将时间恢复到 `15.087 ms`、距 fixed throughput `2.00%`。模型增加 per-worker L2 retention、完整 active stage window 与 completion-feedback gate 要求；production planner 和默认 runtime 不变。 |
| 2026-08-01 | v0.57 | 将 analytical backend 重建为 phase-aware ECM：每个 N range 显式拆为零流量 setup、首 M-panel cold-B 和剩余 steady-B，compulsory packed-B DRAM 只进入 cold phase，所有资源 demand 在拆分前后守恒；并发事件按各资源实际 requester threads 计算 offered-load/capacity/utilization/dilation，并增加同推进器的 `explain_dag()` 诊断。该模型不使用 task-pair slowdown matrix；56 个逻辑测试通过，production 仍等待 8-core/192-core 独立薄校准与 holdout gate。 |
| 2026-08-01 | v0.58 | 在 AmazonC5192Cores NUMA0 完成 matrix/L1/L2/LLC/DRAM 独立薄校准和 96-expert holdout；明确 load probe 为 endpoint service 并按最大下界组合，加入 packed-B repeated-scan 三点 L2 retention，资源诊断增加 allocated rate 容量守恒。108 个 isolated holdout MAPE 为 10.22%，contention P90 为 47.79%，最大 shape regret 为 8.17%，未通过 10/15/5% gate，故 empirical backend 保持 production 默认。 |
| 2026-08-02 | v0.59 | 解析计算 ceiling 改为由 sysfs L1D 容量派生的 M12 L1-hot full-no-store GEMM core service；register-only BFMMLA/frontend/L1 降为诊断和旧 schema fallback，L2/LLC/DRAM endpoint 保持独立上界。AmazonC5192Cores 的 4096-run 1T/96T core ceiling 为 0.340/30.377 TFLOP/s，holdout MAPE 9.18%、contention P90 49.57%、最大 regret 8.17%；仅 isolated gate 通过，production 默认不变。 |
| 2026-08-02 | v0.60 | 在 8-core Neoverse-V1 上复验 sysfs cache-derived 几何：L1D/L2 为 64 KiB/1 MiB，对应 M12/K728/N16 与 M12/K9360/N16；1T/8T L1-hot ceiling 为 0.289/1.621 TFLOP/s，register-only 高估 14.4%/13.1%。记录 L2 长 K 的 2.1%--4.5% 固定成本摊销差异和 V1 无 lower-cache 流量时的 5.60x all-core derate；公式、剪枝和 production 默认不变。 |
| 2026-08-02 | v0.61 | 将 L1-hot full-no-store `gemm_core_flops` 固化为解析模型唯一计算 peak；register-only `matrix_flops` 仅保留诊断，缺少 L1-hot service 的旧 calibration 改为明确拒绝加载，不再使用 kernel 无法达到的 matrix/frontend/L1 fallback。phase 公式、资源取 max 方式、planner 候选和 production empirical 默认不变。 |
| 2026-08-02 | v0.62 | 用整批 native 计时复核 V3 高核数 compute scaling：L1-hot 在 48T 前保持 99.7% 线性效率，64/80/96T 降为 95.8%/93.5%/91.8%，96T 五次稳定在 30.406--30.447 TFLOP/s；register-only 96T 仍为 98.9%。确认高核数 full-loop derate 真实存在但不是 BFMMLA 单元共享，并要求后续用分段 active-core efficiency 代替把拐点扩散到低核数的单一 power curve；当前 calibration 数值和 production 默认不变。 |
| 2026-08-02 | v0.63 | 用 V3 top-down PMU 与 48T victim/aggressor 实验归因高核数 L1-hot derate：L1 refill ratio 不增、L2D stall 为 0、频率保持 3.3 GHz；增长来自 CPU-side backend busy 和 vector issue queue full。48 个 register-only aggressor 只使 full-loop victim 中位时间增加 0.42%，48 个 full-loop aggressor 增加 9.01%。因此模型将其定义为 load-to-vector-compute 混合流触发的 socket-wide vector-dispatch backpressure；最终 firmware power/dispatch 机制因无直接计数器仍不作硬编码，production 默认不变。 |
| 2026-08-02 | v0.64 | 用 M12 双 B 寄存器 column-pipeline 检验局部调度能否关闭 V3 高核数 derate：96T 长窗口吞吐只提升 0.648%，线性效率增加 0.592 个百分点，而 `DISPATCH_STALL_IQ_VX` 增加约 37.5%。该 probe 低于 2% 采用门槛，不进入 production；现有 L1-hot service 和分段 active-core efficiency 结论不变。 |
| 2026-08-02 | v0.65 | 增加纯 GEMM 四状态 shadow 验证：由 M-panel/N-tile 循环精确计数 cold/cold、hot-A/cold-B、cold-A/hot-B、hot/hot，并用 4T 独立冷权重扫 M12--M2040。M>=192 的总误差不超过 4.85%，但 M24 低估 16.53%，证明状态计数可解释而单组 K728 service cost 不能跨 K、cache level 和固定成本直接线性迁移；新增可重复 profiler/validator，不改变 production backend、phase 公式、候选或剪枝。 |
| 2026-08-06 | v0.66 | 把 packed-B 复用的实现条件写成每线程窗口 $\omega=g_s/t$ 与 range 数 $R=\lceil W_s/g_s\rceil$ 的函数：有效重读因子 $p_{\mathrm{eff}}\in[1,\lceil M/12\rceil]$ 由 $\omega$ 主导，加宽 team 与缩小窗口是达到同一 $\omega$ 的可互换路径，但只有窗口会乘 $R$ 的固定成本、只有宽度会降低 memory-level parallelism，因此最优点为内部解。这统一了 8.2.2 的 $Q_{\mathrm{shared},B}=2KNP$ 上界与 9.23 四状态计数的 hot-B 下界。AmazonC5192Cores NUMA0 在 $M=28$ 上实测 $p_{\mathrm{eff}}$ 随 $\omega$ 由 2.97 单调降到 1.22（上界 3 在 $\omega=4$ MiB 取到），同 $\omega$ 下四种宽度差异仅 3.0%--5.6%，$M=12$ 对照极差 0.7%。固定 $t$ 改变活跃核数的分离实验以相反符号否证共享 LLC 容量假设：同一聚合窗口下 $p_{\mathrm{eff}}$ 相差 2 倍并跟随 $\omega$，固定 $\omega$ 时聚合变化 8 倍只变 1.36--1.77 倍且压力越大越好，故 $g(M,t)$ 无需活跃核数项，有效驻留容量约为标称私有 L2 的 1/8。production planner 端到端 A/B 中，$13\le M\le48$ 的候选 band 使 `dsv4-real-2048-seq70` 与 `moe256-uniform` 分别提升 7.36% 与 24.20%，三个无 band 内 expert 的 workload 变化不超过 0.34%。公式、候选空间、宽度剪枝与 production 默认 policy 均不变。 |
| 2026-08-07 | v0.67 | 用 PMU 把 $p_{\mathrm{eff}}$ 从墙钟反推升级为直接测量，并据此把 stage-window policy 的输入单位从每 range 字节改为每线程窗口 $\omega$。`l2d_cache_refill`（含硬件预取）在 $M=12$ 对照上给出跨 L2 字节 / 必需 packed-B $=1.02$，标定了口径；$M=28$、`24x4T` 下扣除 A 与 C 后反解的 $p_{\mathrm{eff}}$ 为 $1.83/1.15/1.16$，与墙钟的 $1.75/1.22/1.23$ 吻合在 $5\%$ 内，故该因子就是 packed-B 在 L2 边界上的重复搬运次数。大 $M$ 的主导项被改写：$M=2040$ 缩窗口使跨 L2 流量涨 $8.2$ 倍且在 $\omega=1$ MiB 处已是必需 B 的 $21.8$ 倍，主体是 shared-A（$2MK_s$，W13 在 $M=2040$ 时为 $16.7$ MB、装不进私有 L2）被每 range 重扫，而非 B 复用率；A 重扫增量预测 $+6.9$ ms 对实测 $+6.18$ ms。参数化上给出 $\omega\to g_s$ 的整数反解（与 8.3 的 $\widehat S_s$ 精确互逆，plan 与 kernel ABI 仍只见整数 $R$），可达 $\omega$ 量化为 $b_s$ 的整数倍、下界 $b_s$、上界 $W_s/t$。已标定的 V1 表由此从 26 个 $g_s$ 塌缩为 8 个 $\omega$ 加 6 个只差一档的偏差格——该表当初按 $(\text{band},t)$ 逐格独立搜索却自行收敛到常数 $\omega$，构成 $\omega$ 为不变量的独立证据。两个 stage 的 $\omega^\ast$ 因 shared-A 相差 $H/F=8$ 倍而不相等，故各保留一个标量。split/no-split 降级为 $g_{W13}\in\{4,8\}$ MiB 的退化情形，identity 记录的 achieved $R$ 无单位、两种编码共享标定数据。production 默认升级到 `amazon_c5_192c_tp4_f512_v2`，新增 $13\le M\le48$ band（$\omega=1/4$ MiB，`widths=(1,2,4,8)`，$t=8$ 取 $1/8$ MiB）：`dsv4-real-2048-seq70` 提速 $7.59\%$、`moe256-uniform` 提速 $24.29\%$，三个无 band 内 expert 的 workload 变化在 $\pm0.33\%$ 而同期 legacy 变体自身摆动 $-0.72\%$--$+0.38\%$。公式、候选空间与宽度剪枝不变。 |
| 2026-08-07 | v0.68 | 用 `--small-w2-window-sweep` 把 $\omega_{W13}$ 与 $\omega_{W2}$ 做叉乘标定（per-task stage window 只在 Plan V2 存在，故该模式改走 `fused_moe_bf16_tiled_async_plan`），在 `24x4T`、192 同构 expert、$M\in\{13,28,48,120,320\}$ 上得到三项结论。其一，$M=28/120/320$ 的二维最优**精确等于**已标定 band 的 $(\omega_{W13},\omega_{W2})$，而那些值当初是按每 range 字节逐格搜索的，构成对标定表与 $\omega$ 参数化的独立验证。其二，$\omega_{W13}$ 是强轴（偏离一档损失 $3\%$--$30\%$）而 $\omega_{W2}$ 是弱轴（固定最优 $\omega_{W13}$ 后极差多在 $1.5\%$ 内，较大极差全部来自 $\omega_{W2}=1/2$ MiB 的悬崖），因此 v0.67 中"两 stage 共用一个 $\omega$ 使收益成为下界"的保留撤销。其三，$H/F=8$ 的 shared-A 论证只在 $M=320$ 成立（$4$ 倍差），$M=120$ 两者相等，$M\le48$ 时 W2 反而偏好更大的 $\omega$——此时两 stage 的 shared-A 均为 $13$--$393$ KiB、都装得进私有 L2，A 重扫项可忽略，短 route 侧的 $\omega_{W2}^\ast$ 机制未识别但效应 $\le1.5\%$。`13--48` band 的 $\omega_{W13}$ 保持 $1/4$ MiB：$M=13/48$ 的孤立最优 $1/8$ MiB 好 $1.36\%$/$2.95\%$，但 $M=28$ 最优为现值，且端到端只在 `moe256-uniform` 上因 cost model 把 shape 从 `8T` 翻到 `1T` 而得 $+1.33\%$（同次运行里固定 shape 的 `manual` 变体反而退化 $0.85\%$），低于 2% 采用门槛且该形状无孤立标定支撑。production 默认、公式与候选空间均不变。 |
| 2026-08-07 | v0.69 | 界定 $\omega$ 不变性的宽度边界并据此补齐 `49--95` band 的窄端。固定 $\omega$ 扫六个宽度显示不变性只在 `1T`--`8T` 成立（极差 $4.2\%$/$11.4\%$）：$M=28$ 在 $\omega=0.25$ MiB 下 `16T`/`32T` 相对 `4T` 峰值掉 $13.0\%$/$27.0\%$，$M=120$ 在 $\omega=0.125$ MiB 下掉 $19.7\%$/$39.0\%$，超过 8 线程后队内同步与并发 expert 数塌到 6/3 的代价超过窗口收益。这不影响 policy，因为 operator-wide legacy 几何本身是 $\omega_{\text{legacy}}=4\ \mathrm{MiB}/t$——`1T` 为 $4$ MiB 而 `32T` 为 $0.125$ MiB，宽 team 从来不在病态区：`16T`/`32T` 上最优窗口相对 legacy 只值 $+2.4\%$/$+0.8\%$（$M=28$）与 $+4.2\%$/$+8.5\%$（$M=120$），故这两个宽度保持继承。真正的空洞是 `49--95` band 只标定了 `8T`：$M=72$ 上 legacy 对最优窗口为 `1T` $3.39\times$、`2T` $2.94\times$、`4T` $1.65\times$，因为 $\omega_{\text{legacy}}$ 在那里是 $4$/$2$/$1$ MiB、比最优大 $32$--$16$ 倍。production 默认升级到 `amazon_c5_192c_tp4_f512_v3`，用实测最优补齐三格（`1T`/`2T` 取 $1/8$ MiB，`4T` 取 $1/16$ MiB），`8T` 逐字节保持 V1 值。catalog 无 preset 会把 $49\le M\le95$ 的 expert 排到 8 线程以下，故 V2 与 V3 在五个 preset 上生成逐元素相同的 plan、墙钟差异按定义是噪声；收益是潜在的，且 cost model 不再用 full-workload anchor 给这三格评分。公式、候选空间与宽度剪枝不变。 |
| 2026-08-07 | v0.70 | 评估并否决"让 split/no-split 退出 planner 候选维度"。8.2.5 指出 `w13_split` 只是 $g_{W13}\in\{4,8\}$ MiB 的布尔别名，但 $\omega$ band 永远不会覆盖全部 $(M,t)$，且其中两项是有意为之：$M\le12$ 时 $P=1$、没有 packed-B 复用可保护（实测窗口效应仅 $\pm0.7\%$），$t>8$ 时 9.26 已实测 $\omega_{\text{legacy}}=4\ \mathrm{MiB}/t$ 接近最优且 $\omega$ 不可迁移；只有 $M>575$ 属于可补但需新标定。V3 在 $(M,t)$ 网格上覆盖 $52\%$，production preset 上仍走 operator-wide 几何的 task 占比从 `moe256-uniform` 的 $0\%$ 到 `moe256-long-short-bimodal` 的 $100\%$（其 route 只有 $\{12,2040\}$、宽度只有 $\{1,16\}$，每格都落在上述原因里），对后者 split/no-split 是唯一的窗口控制。因此 operator-wide 几何不是可退役的兼容层而是所有继承 task 的实际窗口来源，`policy_variants()` 要求完整 legacy pair 的前置条件保留；该 identity 的候选集本身已是最小的两项（split 与 no-split，无 window variant），没有可塌缩的维度。新增 9.27 记录该否证。代码、production 默认、公式与候选空间均不变。 |
| 2026-08-07 | v0.71 | 定位 $\omega^\ast(M)$ 的门限并据此拆分 `144--287` band。8.2.5 预测门限来自每线程 shared-A（$2MK_s$，N 轴分给线程故每线程读全部 A 行）是否装得进私有 L2：W13 落在 $M=256$，W2 落在 $M=2048$，比值即 $H/F=8$。在 `24x4T`、192 同构 expert、$\omega_{W2}=1/8$ MiB 下扫 $M\in\{120,160,176,192,200,224,240,256,300,320\}$，$\omega^\ast_{W13}$ 随 $A_{W13}/L2$ 单调爬升 $1/8\to1/4\to1/2$ 并在 $A/L2=1.00$ 处到顶；起点在 $A/L2\approx0.62$--$0.78$（A 与 packed B、intermediate、C 共享 L2）。门限与宽度无关，符合机制预测：$M=256$ 上 `1T/2T/4T/8T` 全部峰值在 $1/2$ MiB，旧 band 的 $1/8$ 分别损失 $61\%/16\%/23\%/27\%$。这暴露 V1 表的缺陷——band `144--287` 整段用 $1/8$ MiB，而 $M\ge224$ 的最优是 $1/2$ MiB。交叉点由 $M=200$ 与 $M=224$ 内插得 $M\approx217$，production 默认升级到 `amazon_c5_192c_tp4_f512_v4`，拆成 `144--215`（沿用 V1 原值含 `8T` override）与 `216--287`（$(1/2,1/8)$，无 override），边界取 216 即整数个 M12 panel、接缝处最多损失 $0.2\%$；`288--575` 不动。$1/4$ 的"平台"只是 $M=200$ 单点且仅优于 $1/8$ 共 $0.2\%$，不单设 band。catalog 里落在新 band 的只有 `dsv4-real-2048-seq70` 的 3/223 个 task，故五个 preset 端到端差异均在 $\pm0.78\%$、shape 全不变，收益是潜在的。公式、候选空间与宽度剪枝不变。 |
| 2026-08-07 | v0.72 | 把 MoE 与 attention 的大 buffer 分配统一到一个 page policy 层（`csrc/page_policy.h`，header-only 因为 `_moe_C` 不链接 `_C`），并用它实测排除 9.28 门限的 paging 解释。此前分页策略散落三处、且大部分 buffer 未被覆盖：`backend_allocator` 只作用于 `HierarchicalGroupScratch` 的 4 个成员（而 production async 路径用的是全裸 `std::vector` 的 `ScheduledTeamScratch`），attention 的 `WorkspacePool` 硬编码 2 MiB 对齐，packed 权重由 Python 在打包后搬到 hugetlbfs 文件。现在 env 面收敛为 `FUSED_CPP_PAGES=small\|thp\|hugetlb` / `FUSED_CPP_PAGE_SIZE_MB` / `FUSED_CPP_PAGE_MIN_KB` / `FUSED_CPP_HUGETLBFS_PATH`，旧的四个变量作弃用别名；20 个 scratch buffer、workspace pool 与 packed 权重全部经 `page_alloc`。三点实测：其一，hugetlb 下每个 buffer 各自向上取整会把 40 个 scratch 的峰值从 $58.5$ 放大到 $1280$ MiB，故加 `hugetlb_min_bytes`（默认一整页）门限，加后回落到 $112$ MiB。其二，packed 权重改为直接分配后只用 96 个 32 MiB 页（$3$ GiB，恰等于其大小），而旧的打包后搬移需要 $96+96=192$ 页；大页按 NUMA node 分配（每 node 160 页），故旧路径在 `--membind=0` 下于 $E=256/H=4096/F=512$ 规模直接失败——新路径把预算减半并首次让该规模在单 node 上可行。其三，$M\\in\\{192,224,256\\}$ 上 4 KiB / THP / 32 MiB 三种页给出逐格相同的 $\\omega^\\ast$ 与同样的台阶、绝对值差 $\\le0.6\\%$，故 9.28 的门限是 cache 容量效应，A 侧 TLB 混淆以实测排除。稳态性能中性（5 preset delta $\\pm1.37\\%$，legacy 噪声 $-0.67\\%$--$+1.33\\%$）。公式、候选空间、宽度剪枝与 stage-window 表均不变。 |
| 2026-08-08 | v0.73 | 修正 `FUSED_CPP_PAGES=small` 的语义并据此推翻"页大小对连续访存无影响"的结论。原 `kSmall` 走 `::operator new`，落在 torch 已 `madvise(MADV_HUGEPAGE)` 的 arena 内**静默继承 THP**（逐 VMA 实测 AnonHugePages $764/768$ MiB），因此此前的"4 KiB 对照组"与 `thp` 组是同一个东西，$\le0.26\%$ 的差异是伪结果。改为显式 `mmap` + `MADV_NOHUGEPAGE` 后对照组归零（$0/768$ MiB），实测 packed-B 有用带宽 $265.9$（4 KiB）→ $301.1$（THP 2 MiB）→ $304.0$ GB/s（hugetlb 32 MiB）：**连续访存下大页值 $+13.2\%$，且拐点在 4 KiB→2 MiB，2 MiB→32 MiB 只再多 $1.0\%$**。机制为预取器不跨页边界（4 KiB 仅 64 行即重起），非 TLB 覆盖（$2.3$ GB 在 2 MiB 页上为 1150 项，仍在 L2 TLB 约 2048 项内）。9.28 的门限结论不受影响：`thp` 与 `hugetlb` 逐格 argmax 相同、绝对值差 $\le0.5\%$，故整张校准表无需改动；但表隐含"运行于大页"这一前提——$M=28$ 在 4 KiB 下 $\omega^\ast$ 由 $1/4$ 降到 $1/16$ MiB，沿用表值慢 $5.5\%$，而 $M=120$ 的 $\omega^\ast=1/8$ 三种页大小均不变。逐 mapping 诊断（`FUSED_CPP_PAGE_TRACK=1` + `_moe_C.page_mappings()`）另查出 scratch **每次调用重新分配再释放**（`total_allocations` 每调用 $+40$，`live_mappings` 恒为 2），故 Step 2 的"换大页性能中性"应解读为"scratch 从未真正换到大页"，而非"scratch 不在乎页大小"。|
| 2026-08-08 | v0.74 | 新增 9.29：记录 team 分割几何 `kM`/`kN` 及其对 $\omega$ 适用域的限制。`choose_moe_gemm_split` 按 $(stage,M,t)$ 选轴，而 weight-window 循环恒切 N 轴，且**不存在 cache 级 M 分块**。`kN` 下每线程 B 占用为 $g_s/t=\omega$、每线程 A 为 $2MK_s$；`kM` 下每线程 B 占用为 $g_s=t\omega$、每线程 A 为 $2(M/t)K_s$，故 $\omega$ 作为"每线程窗口"**仅在 `kN` 下成立**，而 policy 无条件按 $g_s=t\omega$ 下放。据此修正两处：8.2.5 的 $M=2040$/`24x4T` W13 实为 `kM`，$16.7$ MB 是每 range **全队**量而非每线程量，每线程为 $3.98$ MiB（$A/L2=1.99$ 而非 $7.97$）；该点 $+6.9$ ms 预测与 $+6.18$ ms 实测的吻合**反过来判别了几何**——按 `kN` 算应为 $9.63$ GB $\approx27.5$ ms，差 $4.4$ 倍，故实测独立确认 `kM`。9.28 的门限推导以 `kN` 为前提，其测点（$M\le320$，$t\le8$）除 $M\le32\ \&\ t=2$ 外均为 `kN`，适用域内成立。分割规则切开两个 band：`13--48@t=2`（$M\le32$ 为 `kM`）与 `288--575@t=8`（$M\ge512$ 为 `kM`）；W2 门限过高故 catalog 内恒为 `kN`，同一次调用两 stage 可走不同几何。校准数值不因此改动（实测所得，描述真实行为），改动的是解释与跨宽度外推的合法性：跨越交界的 $\omega$ 外推无效。|
| 2026-08-08 | v0.75 | 新增 9.30：量出 band 表的实际覆盖率并定位 `96--143` 的 band 内梯度。九个 catalog preset 全部经 production planner 规划后按 (route, 宽度) 归格，`amazon_c5_192c_tp4_f512_v4` 的 24 格**只有 6 格被触及**：全部 1T/2T 格为空，4T 仅 `96--143` 一格，其余全在 8T；另有 285 个 task 落表外继承 operator-wide 窗口，含 `moe256-long-short-bimodal` 的全部 179 个。这直接解释了三处 latent gain（9.26 的 `49--95` 窄格、9.28 的 `216--287`、9.26 保留的 16T/32T 继承）——A/B 只能报噪声不是因为标定错，而是没有 preset 会排到那里，故补格子前应先查覆盖。同时暴露 A/B 的结构性盲区：唯一活跃的 4T 格所属 preset 在 A/B 集合内，但两臂为 $1$ MiB 与 $0.0625$ MiB，胜过两者的 $0.125$ MiB 不在任何一臂中。`96--143` 用单一 $\omega$ 覆盖 48 个 route 而最优值在 band 内迁移：4T 上 $\omega=0.125$ 相对生产值 $0.0625$ 在 $M=96/108/120$ 分别领先 $0.28\%/1.44\%/3.26\%$，交叉点在 $M=96$--$108$ 之间，而 9.28 门限表里 $M=120$ 峰值在 $1/8$ 而非生产用的 $1/16$ 这一矛盾此前未标注。唯一活跃格 `moe256-active-set-128` 的 128 个 expert 全在 $route=96$ 即交叉点上，损失 $0.28\%$ 低于噪声底，故不改值；记录为已知偏差：$100$--$143$ 的 4T workload 付 $1.4$--$3.3\%$，该格只在下沿验证过。|
| 2026-08-08 | v0.76 | 补齐 8 点 $M$ 网格并撤回一处非单调读法；新增 9.31（残余代价不在任何已测计数器中）与 9.32（barrier 是线程饥饿的阶跃）。固定 $g_{W2}=0.5$ MiB 后 4T 上的最优 tile 数是**单调阶梯 1、2、8**：$M\le40$ 窗口无关（4 倍窗口变化只动 $0.33$--$0.59\%$，在 $0.5\%$ 重复性内，故 `13--48` 的 $0.25$ MiB 是三个不可区分选项之一）、$M=56$--$96$ 为 1 tile、$M\ge108$ 为 2 tile、$M\ge224$ 为 8 tile。此前把子集读成"$4\to1\to2$ 下凹"是混用了不同 $g_{W2}$ 的伪像；由此发现二阶耦合——**W2 窗口改变 W13 曲线形状**（$M=28$ 上 $g_{W2}=0.25$ 时 W13 有 $2.0\%$ 梯度，$0.5$ 时变平），9.25 的"W2 为弱轴"按其自身极差仍成立但两轴形状不独立。另两个压在 1-tile 下界的格子实测生产值正确（`49-95@4T` 优 $2.0\%$、`96-143@2T` 优 $0.27\%$），故 `96-143@4T` 是单一 band 边界个案而非系统性偏小；台阶在 $M\approx100$，活跃 preset 恰在 $route=96$ 处两窗口仅差 $0.07\%$，不改值。$\omega$ 的下界经代码与 profile 双重确认为**恰好一个 tile**（$K\cdot\text{n\_tile}\cdot2$，W13 $64$ KiB、W2 $8$ KiB，`max(1,\cdot)` clamp，tile 不可再分），越界的代价是多余线程零工作量后在 barrier **自旋**：占比随空闲线程比例阶跃（$3/4$ 空闲 $41.78\%$、$2/4$ 空闲 $21.86\%$、不空闲 $1.2\%$）而与 $R$ 无关（$R$ 变 8 倍不动）。自旋以高 IPC（$4.76$ vs $3.62$）退休指令，计入 `instructions` 而不计入 `stall_backend`，故越界窗口在 profile 里表现为"多做工作"而非"多等待"——这曾使我先后误判为同步开销、micro-kernel 摊销、A 重载三次。合法区内的残余代价（$M=72$ 上 $5.4\%$）则五个计数器全平，逐一否证 L2 容量、A 重扫流量、B 重读流量、指令数、barrier 与并发干扰，指向延迟/重叠而非体量，与页大小那 $13\%$ 同类，待用需求-预取拆分判别。|
| 2026-08-09 | v0.77 | analytical backend 新增公式生成的 per-task W13/W2 stage-window policy：从 tile-aligned 可达集中过滤 L1D 以下窗口、线程饥饿和非二分 owner tile，W13/W2 分别最小化串行增量 ECM 目标；绝对时间继续使用原 ECM maximum/event simulator，planner 搜索空间不增加。machine schema 增加 runtime `backend_n_tile` 和独立 packed-B retention 有效容量，修正文档中旧 `kM/kN` 门限为历史行为（当前 production 固定 `kN`）。两机 interleaved holdout 中，192C median/P90/max regret 为 `1.63/6.57/11.32%`，8C raw 为 `1.39/3.51/15.22%`、p10 敏感性为 `1.30/1.93/2.78%`；8C retention 仍是显式 transferred prior。clean 192C 未过 5% max gate，故 analytic model 自动使用公式 policy，但 empirical production V4 与候选/剪枝保持不变。 |
| 2026-08-09 | v0.78 | 修正 analytical stage-window cache traffic：A 改为保留一个在途 panel headroom 的物理 resident/streaming 二态；B 的三点 repeated-scan miss 只作用于 $\lfloor C^A_{2,eff}/A_p\rfloor$ 个 transient reuse，之后按 owner stripe 是否超过物理 L2 进入 resident/streaming steady state，避免长 route 把短程残余 miss 乘到全部 panel。选择器将 $\epsilon_{rel}T_{xfer}$ 内的目标视为不可分辨，并在等价集内取最接近 $\max(C_{L1D},2b_s)$ 的 kernel-native 窗口；没有新增 route 表、候选或剪枝。相同 interleaved holdout 上，192C median/P90/max regret 从 `1.63/6.57/11.32%` 降至 `1.50/2.98/3.38%`，28/28 通过 5% gate，median rank rho 由 0.853 升到 0.903；8C raw 仍受强抢占，p10 median/P90/max 为 `2.01/2.61/4.21%`，且 retention 仍为 transferred prior。analytic backend 默认升级到 policy v2；empirical V4 production 和 planner 搜索空间不变。 |
| 2026-08-09 | v0.79 | 开始移除 split 语义：analytic stage-window policy v3 将 W13 `no-split/split` 归一为显式 $R=1/R=2$ 几何端点，与 inherited 和解析自然窗口在同一目标中评分，不再用旧 split chunk 数限制 $R\ge2$；$M\le12$ 因单 panel 对 B 无重用而由支配关系固定取 W13/W2 $R=1$。候选仍由已选 $(M,t)$ 确定性生成，不扩大 planner shape 空间；empirical profile、Plan V2 ABI 和 native runtime 本步不变。holdout 默认覆盖 M12 与 `1/2/4/8T` 并显式标记两个 W13 端点；v3 实机 gate 留待 range ABI/profile 迁移后刷新。 |
| 2026-08-09 | v0.80 | Plan V2 与 ARM native runtime 增加 per-task exact $(R_{13},R_2)$：production planner 在确定 task 的实际宽度后解析正整数 range，并将其用于 W13、W2 GEMM 和 W2 owner-scatter；不增加 shape/window 搜索维度。迁移期 `-1` 继承与 byte-window 字段只兼容旧手写 plan，同一 task/stage 同时指定 byte/range 会被 Python/native 双重拒绝。profile/catalog identity 和公开 split API 留待后续独立步骤迁移。 |
| 2026-08-09 | v0.81 | empirical/analytic profile、catalog、plan cache 与 TP/EP companion matching 的 kernel policy identity 统一为 `("stage_ranges", R13, R2)`；旧 split/byte 字段仅用于 schema-v2 历史 profile 的几何重建，不参与匹配。catalog 不再要求完整布尔 pair，改为接受同 implementation 下任意非空实测 range 集合，同时严格拒绝重复 range identity 和不同 route/thread/shape grid；`auto` 因此可使用只有一个 range variant 的 JIT 校准，但仍禁止跨 JIT/static 拼表。公开 API 与 native legacy fallback 留待下一步删除。 |
| 2026-08-09 | v0.82 | 公开 Python/C++ MoE ABI 与 Plan V2 runtime 收敛为 exact stage ranges：同步、scheduled、legacy async 使用正整数 `w13_ranges/w2_ranges`，Plan V2 必须逐 task 携带正整数 range；legacy bridge 显式升级为 `1/1`。删除布尔 split、全局/per-task byte-window、对应环境变量和 Plan V2 到 legacy async 的 fallback；W13/W2 GEMM 与 W2 owner-scatter 只消费同一份精确 range。kernel layout、N-split ownership、planner shape 候选与解析 stage-window policy 均不变。 |
| 2026-08-09 | v0.83 | 解析/phase cost model、profile catalog、interval planner、TP/EP companion matching 与 contention profile 生成器全部改为 range-only policy：基线由显式正整数 $(R_{13},R_2)$ 构造，内部 cache 字节目标只作为物理模型输入并在进入 plan 前量化；schema-v2 profile 必须显式记录两个 range，旧 split/byte 字段只可作为未读取的历史 provenance。删除 model/query/result 中的 split 别名和 byte identity，不改变 shape 候选、event simulator、stage-window 解析规则或 runtime 执行几何。 |
| 2026-08-09 | v0.84 | 异构 overlap、working-set search、working-set owner-cache 与 $T_{iso}$ roofline 校准工具迁移到 exact $(R_{13},R_2)$：MiB sweep 先按 packed tile 几何量化再调用 runtime，并同时记录请求字节与 achieved range；working-set 尺寸改为两个 stage 最大精确 range，因而不再限定 $R_{13}=2,R_2=1$；$T_{iso}$ 的 W2 shared-A traffic 补乘 $R_2$。工具不再设置 split 环境变量或调用 byte-window ABI，不改变 production planner 候选与剪枝。 |
| 2026-08-09 | v0.85 | benchmark、timeline trace、stage breakdown 与 roofline caller 全部改为 exact $(R_{13},R_2)$：Plan V2 trace 逐 task 记录实际 range，GEMM throughput 标注直接按该 range 还原 owner N 列；字节窗口 sweep 仅保留为校准输入，并在调用 runtime 前量化为 exact range。补齐全部 schema-v2 历史 profile 的显式 range identity，删除校准 helper 的 runtime byte-window compatibility API；旧 split 字段和 profile 文件名仅保留为未读取 provenance。planner 候选、解析窗口目标和 native 执行几何不变。 |
| 2026-08-09 | v0.86 | working-set shadow 的诊断 schema kind 从旧布尔策略命名改为 `stage_range_working_set_band_validation`；其输入、owner-cache 公式、推荐结果和 production 隔离边界均不变。 |
| 2026-08-09 | v0.87 | 文档、Plan/profile schema 与 optimization manifests 收敛到 range-only 契约：公开 entrypoint 只记录正整数 `w13_ranges/w2_ranges`，Plan V2 每 task 必须携带正整数 range，profile 必须显式记录 exact identity；byte target 仅是解析模型在 plan lowering 前的校准输入。修正 homogeneous full-call anchor 的规则为“所有 task 保持 profile baseline pair”，并将默认 stage policy 标记为 V4 的 `R13=2,R2=1` 精确 gate。历史 JSON 字段、profile 文件名和 changelog 术语只作为 provenance 保留。 |
| 2026-08-09 | v0.88 | 删除 operator-wide $(R_{13},R_2)$ planner 控制：`ProfilePolicy/ProfileQuery` 不再包含 range，catalog 每个机器/拓扑/shape/implementation/source 域只允许一个活动校准，旧 `(1,1)/(2,1)` 配对表从活动目录移除；range 字段降级为采样几何 provenance。删除 `PolicyAwarePlanner`、range variant 枚举、联合 `(range,shape)` 搜索、plan-level `operator_options` 和对应 cache identity。planner 只搜索 shape，随后由 $g_\theta(M,t)$ 生成 Plan V2 的逐 task exact range；native cold planner 中的 pair 仅是未覆盖 task 的 calibration fallback，不是候选。同步更新 TP/EP evaluator、validator、schema 和 full-call anchor 判定。 |
| 2026-08-09 | v0.89 | 完全删除 production weight split/range/window 语义：Python/C++ ABI、Plan V2、ARM/x86 executor、native planner、empirical/analytic cost model、profile generator 和默认 benchmark 均只执行 `full_n_team_stripes`。W13/W2 stage bytes 固定为 $4HF/2HF$；team width 通过 $u_s(t)=2K_s\nu\lceil(N_s/\nu)/t\rceil$ 唯一决定每线程 owner stripe。删除逐 task range 张量、解析 stage-window policy、range 搜索/缓存 identity 和旧 active split profiles；route/M slicing 保留且与 weight split 明确区分。profile catalog 拒绝非 full-N 几何，planner 候选/剪枝只保留既有 width、shape、tail-pool 和 bounded route-slice 维度。 |
| 2026-08-10 | v0.90 | 把每线程 owner 窗口恢复为一等参数（9.35）：一个 stage 的计算模式由 $(t,\omega_s)$ 唯一确定，$\text{range}_s=t\omega_s$、$R_s=\lceil q_s/(t\omega_s)\rceil$，尾窗口可短并按 `split_evenly` 分配；$\omega_s=\lceil q_s/t\rceil$ 即 v0.89 的 full-N team stripe，故新参数化是旧几何的**超集**、默认逐位一致。ABI 以 **tile 数**承载（Plan V2 新增 optional `task_w13_window_tiles`/`task_w2_window_tiles`，$0$=全条带），字节换算只允许在 `FullStageGeometry.window_tiles_from_bytes` 一处发生，因为 $\lfloor b/b_s\rfloor$ 是多对一、无法反推唯一模式；与 route/M slicing 的 `task_range_granularities` 正交。三条结构性质：窗口是**纯调度旋钮**（每 (panel, window) 对访问一次、线程写 disjoint C、融合 W13 无 K 分块 ⇒ 任何合法 $\omega$ 逐位相同，已用 rows/宽度/后端矩阵实测）；饥饿边界可判定（$t\mid q_s$ 时尾窗口必为 $t$ 的倍数，TP4 下 $t\in\{1,2,4,8,16,32\}$ 全安全）；下界恰为一个 tile 且与 $t$ 无关。$R>1$ 时禁用 `first_panel_prefetch` 与 `bulk_m`（两者都假设一次调用覆盖整条 stripe）。实测全条带在窄宽度代价很大——$t=4$ 慢 $75\%$--$104\%$、$t=8$ 慢 $23\%$--$31\%$、$t=16$ 在 $M\le120$ 慢 $4\%$--$15\%$，机制是聚合而非单核（$t=4$ 时 96 核共 192 MiB packed B 对 96 MiB L3）；但 $t=16$ 且 route $\ge144$ 起全条带最优、强制 1 tile 在 $M=384$ 慢 $80\%$，与 9.28 大 $M$ 需大窗口摊销 $A$ 重扫的门限方向一致，故 policy 不在该区加 16T；$t=32$ 的 stripe 已只有 4 tile、所有窗口在 $0.6\%$ 噪声内，故意不覆盖。policy 为 route band + per-width override，在已选定 $(M,t)$ 上确定性读表，**不增加 planner 搜索维度**；未覆盖 route/width 返回全条带并继续报 `stage_geometry=full_n_team_stripes`，旧 profile 与 catalog 拒绝规则均不变，仅 $R>1$ 报 `windowed_team_stripes`。planner plan 上端到端 A/B：`moe256-uniform` $+12.4\%$、`dsv4-real-2048-seq70` $+10.6\%$，另两个 preset 在 $1.3\%$--$1.7\%$ 噪声底内。harness 已对齐退役前网格（9 格 $-1.82\%$--$+0.12\%$、argmax 逐格复现）。未验证：`moe256-active-set-128` 落在孤立 $+9.2\%$ 格子却端到端不动的原因；16T 项只标定 W13 轴。|
| 2026-08-10 | v0.91 | 删除已退役的 SVE JIT `first_panel_prefetch` 主路径：移除 W13/W2 环境变量、dispatch 选择、Xbyak 预取生成器与缓存维度、正确性测试、benchmark variant 和 manifest feature；普通 exact-M 与实验 `bulk_m` 的计算语义不变。历史报告和 TODO 继续保留结论：低并发单 panel 可受益，但约 18--20 条冷 B stream 后交叉，高并发与可复用 B 的长 route 无稳定 E2E 收益并可回退。stage-window 当前只需在 $R>1$ 时禁止 stripe 级 `bulk_m`。|
| 2026-08-10 | v0.92 | 按“保留实验知识，不默认保留实验源码”退役性能中性的 SVE JIT `bulk_m`：删除生产生成器 outer-M 状态机、callee-save GPR、cache-key 维度、环境变量、预热/dispatch 分支、重复正确性路径和 benchmark selector。普通 exact-M 继续由 C++ panel 循环逐 M12 调用，公开 API、packed layout、默认 dispatch 和窗口语义不变；历史实现固定在 Git `3faf244`，完整结果继续保留在 `amazon_192c_xbyak_bulk_m.md`。|
| 2026-08-10 | v0.93 | 删除已被 2% gate 否决的 M12 双 B 寄存器 column-pipeline 实验源码：移除两个 JIT probe mode、生成器分支、cache slot、通用 pure-GEMM benchmark selector、专用 scaling benchmark 和重复测试项。production M12 state machine、probe API 的其余模式和 cost-model 校准语义不变；历史实现固定在 Git `4825bf9`，性能与 perf 结论继续保留在 `amazon_192c_m12_column_pipeline_20260802.md`。|
| 2026-08-10 | v0.94 | 将 SVE JIT service probes 从一次性指令分解收敛为薄机器校准接口：删除 BA-only、BFMMLA-only、A-only、control-only、with-store、BA-fixed-A 和 full-no-store-fixed-A 七种模式及专用 Python/perf harness，只保留 profile 实际消费的 B-only、M12 full-no-store 和 M12 matrix-only。三个保留模式继续使用历史 id 1/4/10，避免破坏 calibration provenance；JIT cache 从 11 个稀疏 slot 收敛为 4 个显式 service slot。旧分解实现固定在 Git `9aa9248`，结果保留在 `amazon_192c_small_m_gemm_bandwidth.md`。|
| 2026-08-10 | v0.95 | 退役 M1/M2 dual-N JIT：删除环境变量、双 N-tile K-loop/epilogue、generator/cache 维度、重复正确性分支和 benchmark selector。该路径仅在 8 个 active experts 时约 +2.3%，24-way 回退 5.4%--5.7%，96-way M2 回退约 5.0%；根因是 wave barrier 下 lane-time 方差放大，安全选择需要 planner-visible concurrency 与 barrier topology，超出 kernel-local selector 的合理边界。普通 exact-M state machine、ABI、layout 和默认 dispatch 不变；历史实现固定在 Git `f671f7e`。|
| 2026-08-10 | v0.96 | 退役 cold-phase timed task release：从 Plan V2 schema/materialization、Python bridge、C++ ABI/pybind、native claim loop、strict-tail gate、interval planner 默认字段、CP-SAT runtime lowering、测试和 benchmark 中删除 `task_release_ns`。离线 CP-SAT 的 `ColdPhaseRuntimeTask.release_ns` 继续作为 placement 分析数据，不再声称可由 runtime 复现。该路径使 DSV4 从 14.854 ms 回退到 20.869 ms（-28.82% throughput），且 0.25x release 与 eager mixed 仅差 0.03%；历史实现固定在 Git `94e977d`。|
| 2026-08-10 | v0.97 | 删除 legacy hierarchical N-split 的 `FUSED_CPP_MOE_FUSED_SHARED_APACK` 实验分支。该路径仅在显式关闭 fused packA 时可达，SVE 主路径始终强制 fused packA；历史实验已确认一次性共享 pack 写入相对重复 cached-A 读取和 GEMM 计算无可测收益，却额外引入 pack work 与两个 barrier。默认执行、Plan V2、kernel ABI 和 planner 搜索空间不变；历史实现固定在 Git `a579b24`。|
| 2026-08-10 | v0.98 | 退役 W13-to-W2 elastic cohort runtime：删除专用 C++ scheduler/ABI/pybind、Python execution mode/统计接口、planner bridge、benchmark CLI 与活动测试，Plan V2 明确拒绝 `execution_mode=elastic`、`task_resize_timeout_ns` 和 `task_preferred_core_begins`。通用 workload 回退 2.5%--24.8%，唯一显式 active-set tail 迁移提升 3.66%，但该 planner-visible case 已由 bounded tail repartition 覆盖；历史实现固定在 Git `0b58091`，公式与结果报告继续作为历史知识保留。|
| 2026-08-10 | v0.99 | 退役 production split-K/Kc 配置面：SVE packed-B 固定为 one-chunk `Ntile -> K4`，删除 `FUSED_CPP_MOE_SVE_KC`、L1 比例 selector、Kchunk-major pack、stage-window 几何告警、JIT fallback 判定及专用生产测试。隔离 K-block 在 192C 单核可提升 6%--13%，但 24x4T 主波次 route=12/2040 分别回退 9.52%/1.25%，默认已于 2026-07-19 回到 one-chunk；独立 microbench/asm 和完整报告继续保留，历史 runtime 固定在 Git `44dbcd9`。planner 的 $(t,\omega_s)$、候选与 cost model 不变。|
| 2026-08-10 | v1.00 | 收敛已启用的 SVE expert barrier-elision 语义：删除 `FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO`、`FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER`、legacy barrier baseline、两个单特征 candidate 和四组合 benchmark 源码。SVE fused-SiLU 固定由 W13 完整覆盖 packed intermediate，并由 W2 N-owner 直接 scatter；非 SVE/非 fused 路径继续保留必要初始化与 barrier。96T route=12 的 combined scheduled/async 收益为 7.77%/9.04%，长期默认已覆盖 dirty-scratch、FP32/BF16 route 和全部 M tail；历史比较实现固定在 Git `b6d168d`。planner、cost model、公开 ABI 与默认数值不变。|
| 2026-08-10 | v1.01 | 收敛 SVE weighted route merge 为唯一 U1 production kernel：删除 sequential SVE compatibility、U2/U4 hidden-axis unroll、环境变量/legacy alias、重复 E2E selector benchmark 和跨 variant 测试。fixed TopK 2/4/6/8 tree、dynamic TopK ordered loop、FP32/BF16 route source 和非 SVE scalar fallback 保持不变；完整性能/PMU 结论保留在结果文档，历史实现固定在 Git `7fc10fc`。planner、cost model、候选与数值契约不变。|
| 2026-08-10 | v1.02 | 收敛 M12 fused-SiLU epilogue 为 optimized exact-FDIV 实现：删除旧逐行常量加载 epilogue、NR1/NR2 reciprocal refinement、minimax3 polynomial、模式分支/导出符号/环境变量和对应兼容测试。poly4/5/6 精度选择、M8/M4/M2/M1 tail、identity 诊断、JIT exact-M 与 Lab 使用的 optimized M12 符号保持；NR 路径实测慢 `0.45%--1.37%`，minimax3 相比已有 poly4 只多 `0.07%--0.15%` 且误差更高，历史实现固定在 Git `233436a`。planner、cost model 与调度候选不变。|
| 2026-08-10 | v1.03 | 退役 production W13 identity 诊断：删除 `FUSED_CPP_MOE_W13_SKIP_SILU`、六个 static-asm identity epilogue/导出符号及 stage-profile 元数据。该旁路会改变 fused operator 语义，且不被当前 cost model 消费；后续增量 GEMM/SiLU 分解使用 standalone pure-GEMM/service probes。历史实现固定在 Git `8bac39a`，既有性能与精度结论继续保留。planner、模型公式、候选空间和默认数值不变。|
| 2026-08-10 | v1.04 | 完成 upstream M8/M12 ILV 退役：从活动 pure-GEMM benchmark 删除 `--include-ilv`、ILV 符号适配、第三路输出与三路交错计时，仅保留 JIT 对 non-ILV upstream 的校准比较。ILV 跨机器实验未过 2% gate，精确复现固定在 Git `fe0ca5a`；外部 upstream 汇编文件不被视为 fused-MoE 活动 variant。同步修正 TODO 中已删除 first-panel-prefetch 开关的陈旧保留说明。planner、cost model 和 production dispatch 不变。|
| 2026-08-10 | v1.05 | 修复优化治理索引与当前 window 模型文档：为 5 个 legacy retired feature/variant 补齐最后活动 Git 定位，并将 README/TODO 中“无 window/full-N 仅由线程宽度决定”的陈旧描述改为当前唯一的 $(t,\omega_s)$ 参数化；旧 split/range-count/byte-window 仍保持删除，Plan V2 的 per-task tile window 为零时回到 full owner stripe。仅修正文档与 provenance，不改变公式、planner 或 runtime。|
| 2026-08-10 | v1.06 | 退役 production `FUSED_CPP_MOE_FUSED_2D_SPLIT` compatibility adapter：删除恒定返回 $t_M=1,t_N=t$ 的 2D plan/range structs、NEON/SVE `_2d` wrappers、normal/scheduled/async/planned-staged 分支和重复测试。该实现从未切 M，SVE wrappers 还强制 `row_begin=0`，所以其行为只是已有 N-split 的重复封装；真正的 mixed-MN 假设继续由隔离的 `bench_mn_split.cpp` Lab 路径验证，历史 production 适配器固定在 Git `8e9fcbd`。hierarchical N-split、Plan V2 $(t,\omega_s)$、planner 候选、cost model 公式和默认数值不变。|
| 2026-08-10 | v1.07 | 删除 production `FUSED_CPP_MOE_STAGE_WINDOW_TILES` 实验环境适配器及其 process-static parser。stage window 继续由 Plan V2 的逐 task `task_w13_window_tiles/task_w2_window_tiles` 唯一表达，legacy scheduled/async 与 hierarchical fallback 明确使用 full owner stripe；校准 sweep 通过构造显式 Plan V2 复现，不再让进程环境隐式改写未指定窗口。历史适配器固定在 Git `208743e`。窗口计算语义、planner policy、cost model 公式、候选和默认 Plan V2 数值不变。|
| 2026-08-11 | v1.08 | 闭合解析 stage-window 的当前选择问题（9.36）：shadow policy v6 按 exact tile-window geometry 计算 A/B/C task-local demand，仅以 reusable packed-B 形成 rank 级 LLC resident surcharge；增加 fused-W13/pure-W2 的 per-panel extra-range 薄校准（45.27/6.70 ns），并在模型不确定性等价集内用 $\max(C_{L1D},2b_s,\sqrt{C_{L1D}C^{B,eff}_{L2}})$ 做无 route 表 tie-break。同步实验否证“纯 GEMM restart 足够”和“barrier 决定 M216 转折”两个假设。AmazonC5192Cores NUMA0 当前 tile runtime 的 6 组 full Cartesian holdout 得到 median/P90/max regret `1.10/3.45/4.40%`，6/6 通过预设 5% gate；但极端 pair 的二阶交互残差仍达 17.69%，故只关闭声明域内的 window 选择，不声称绝对时间、跨机器或完整 planner 闭合。production band policy、shape 搜索、剪枝和 Plan V2 默认不变。|
| 2026-08-11 | v1.09 | 增加 benchmark-only large/small route 双资源区：按一个 route threshold 将单 NUMA 核心静态分成两个不重叠区域，区内使用已校准固定宽度和独立 LPT lane，完整复用 Plan V2、stage-window 与 merge 语义。DSV4 TP4/F512 上 `48C x 8T` 大区加 `48C x 1T` 小区在 51 轮中相对 strict `12x8T` 提升 14.99%，trace 两区结束只差 0.214 ms，PMU memory-stall cycles 降 29.07%；但当前 simulator 高估 wall time 28.81%，故该候选不进入 production 搜索、cache identity 或默认执行，待 cross-class phase service 模型与跨 workload/machine gate 闭合。|
| 2026-08-11 | v1.10 | strict planner 增加固定 LPT lane membership 的时间交错顺序：每个 shape 只比较原 LPT、奇数 lane 反序和偶数 lane 反序三个 deterministic DAG，以完整 event-time model 严格判优；full-call anchor、线程宽度、window、core interval 和 Plan V2 runtime contract 不变。当前 TP4/F512 profile 对 captured DSV4/tiered hotspot 的 strict-only 预测改善为 10.01%/8.52%，active-set-128 保持 LPT；这些仍是模型预测，cross-class 实机 gate 未闭合。Python/native parity、合成 contention 改善与 tie-retention 已覆盖；`PlannedMoE` 缓存三态 order 并在命中时跳过 DAG 重评分。逐 lane 坐标下降因收益不足且 cold-plan 成本过高而未纳入。|
| 2026-08-11 | v1.11 | 在 AmazonC5192Cores NUMA0 对 fixed-lane temporal order 做 7-warmup/51-pair 真实交错 A/B。strict captured DSV4 为 `+8.34%`，默认 tail-pool 为 `+10.65%`，独立 seed 复测 `+10.64%`；但 tiered 为 `-0.35%`，strict long/short bimodal 回退 `-5.90%`（tail-pool 下 `+0.11%`）。uniform 相同 bridge 的 paired noise 为 `-0.01%`，全部输出逐位一致。由此确认模型能选中 DSV4 收益，但 cross-class 时间排序尚未闭合并违反 2% held-out gate；第一版不能无条件进入 production strict 顺序，需先完成 stage trace 和保守启用条件。|
| 2026-08-11 | v1.12 | 定位 temporal strict bimodal 回退：phase trace 显示 reverse-even 的 compute end 比 LPT 提前 `0.80--1.05 ms`，W13/W2 与 compute core-ms 都更低；但 compute 后 ready token 从 `47/59` 增至 `2040/2041`，fixed-owner merge tail 从 `0.037/0.051 ms` 放大到 `0.347/1.939 ms`。原因是最后长 expert 的单个 `local_tid==0` 串行扫描约 2040 routes 并发布 token，同时 96 owners 轮询/排空，形成当前 expert-only DAG 未建模的 burst 与双稳态。关闭 ready-token early merge 后，两个独立 51-pair strict A/B 稳定提升 `+5.88%/+6.07%`，证明回退来自 merge-readiness 交互而非 GEMM 时间交错。production gate 继续保持 open，下一步需联合建模 token readiness 或对 bursty temporal plan 禁用 early merge。|
| 2026-08-11 | v1.13 | strict planner 增加不扩大搜索空间的 routing-shape/route-bound early-merge 保守 gate：复用选中 compute DAG 的 expert finish time；对每个高覆盖 expert，以自身 route count 减去所有更晚波次 route occurrence，形成其 ready burst 的安全下界。若该下界之外的 token 不超过默认一轮 fixed-owner drain（$B_{merge}T=2T$），Plan V2 写入 `early_merge=false`，否则保持 `null/auto`，从不强制开启。gate 只读取 `topk_ids` 的 $(N,K)$ 形状，复杂度 $O(E|\mathcal H|)$，不扫描 token 内容；冷/热 path 每次重算，不把 token 数决策错误缓存到 histogram shape 上。tail-pool、candidate score、shape cache、native planner ABI 与 kernel ABI 不变。合成测试覆盖同 histogram 的不同 $(N,K)$ 在 cache hit 上 `null -> false -> null`；plan-only 对 selected strict `6x16T` bimodal 得到 `2016/2048` burst 下界并关闭，对 DSV4/tiered 的 `917/384` 保持 auto。AmazonC5192Cores NUMA0 的 9-case strict catalog 以 7-warmup/51-pair 复验得到 bimodal/DSV4/tiered `+6.28%/+8.80%/-0.56%`，六个未改序对照 median 绝对偏差不超过 `0.89%`，关闭该 host 的 2% held-out gate。|
| 2026-08-11 | v1.14 | 增加 benchmark-only large/medium/short bounded-stream comparator：在原 `48C x 8T + 48C x 1T` shape 内，为 M<=12 静态保留可配置数量的 1T lane，并让三类 LPT 链从调用起点并发。DSV4 上 isolated short service 预测 3 lane 足够，但 4/6/8 lane 实测严重成为尾部；12--16 lane 才恢复，51-round 最优 16 lane 为 11.789 ms，仍比原双区 11.595 ms 慢 1.67%。trace 中 short 启动从 5.177 提前到 0.277 ms、峰值 26 降到 16，但 short W13+W2 core-time 不降，medium W13 增加 5.6%，且 short 区提前约 2 ms 空闲。delayed-short 控制证明 service time 随 active mix 显著变化，但因同时改变活跃核心数、wave 和尾部，不能完成因果分解。进一步固定同一批 M28/M1、48 个 1T lane 与每核任务量；独立 long-expert 进程只制造 0/16/32/48 核背景而不进入前景计时。背景使 grouped 前景回退 20.0%/36.8%/48.0%，证明争用真实存在，但 grouped/crossed 无单调差异：16 核交错略慢、48 核中性，仅 32 核出现约 1% 且置信区间跨零的弱收益。因此均匀 M/S 混合不进入 planner 约束，只可作无成本 tie-break；新入口仅用于后续 allocation/release/tail 对照，不进入 production planner、cache identity 或默认 runtime。|
| 2026-08-12 | v1.15 | 将默认关闭的 strict suffix-steal 从“全 plan 单一宽度、单一 NUMA”泛化为 `(threads, NUMA)` cohort：runtime 从 fixed whole-expert task 恢复不重叠 team 分区，只在同 cohort 领取尚未启动的受限后缀，禁止改宽、跨 NUMA、抢占或等待；scratch lease 前按 cohort 最大 route 扩容，迁移期间不分配。新增混合 `2T+1T` 双 cohort SVE 回归，要求两个 cohort 均实际迁移且输出逐位一致。opt-in 的 $r_{\min}$ 从 2 改为 1，使存在一个完整 pending task 时即可领取；feature enable 默认仍关闭。AmazonC5192Cores NUMA0 的 DSV4 `48C×8T + 32C×1T medium + 16C×1T short` 做 11 warmup/101 paired runs：strict/candidate 中位数为 `11.738/11.720 ms`（`+0.15%`），配对均值 `+0.20%` 的 95% bootstrap 区间为 `[-0.21%,+0.60%]`，P90 由 `11.863` 增至 `11.910 ms`；候选 trace 迁移 `20` 个 expert（`19×1T + 1×8T`，686 routes），tail idle 降至 `75.1 core-ms`，但未通过 2% E2E gate。因此 planner candidate、cache identity、cost-model 评分保持不变，不恢复已退役的 W2 boundary elastic。|
| 2026-08-12 | v1.16 | 增加与 production 解耦的第一版可证明 makespan 下界：通用层实现逐资源/critical-chain 的 `LB0` 和共享 fractional mode 的 mode-relaxed LP，并输出经整数 simplex 量化、`Fraction` 精确重算和定向舍入的 dual/primal certificate；优先使用 GLOP，缺少可选依赖时回退 entropic mirror ascent。SVE adapter 将当前 exact-M W13/W2 mapper 降为 BFMMLA、key instruction、L1 load、epilogue、core-time 和可选 compulsory-DRAM 需求；core-time 由 aggregate work/per-core ceiling 推导，不假设不均衡 N lane 等时结束。第一版省略 window replay、gather/merge、NUMA 与 contention，只提供安全但偏松的 GEMM-only 下界，不改变 production 候选、剪枝、cost model 或 runtime。|
| 2026-08-12 | v1.17 | 闭合四类 DSV4 尾部候选：完整 `1/2/4/8T` cohort 重组、非阻塞 `2/4T` 重组、短任务队列重排、cost-model suffix DAG 和实际完成时 residual-M 均为中性或负收益，相关 runtime/schema/API/planner 实现全部删除。唯一稳定为正的是手工 E218/core40 静态 residual-M route slice，五次复测提升 `0.48%--1.20%`（均值约 `0.93%`），但其他 terminal target 为 `-0.19%--+0.05%`，且未过 2% production gate；仅保留复用现有 Plan V2 的 Lab benchmark comparator，不改变默认 planner、cost model、cache identity 或 production runtime。|
| 2026-08-13 | v1.18 | analytical machine schema 升到 v2、model schema 升到 v7：从 Linux `shared_cpu_list` 显式记录 rank CPU 与 LLC 域分区/容量/域内 refill curve；LLC/DRAM 服务改为 isotonic 去噪后的实测点分段插值。给定 placement 时 LLC 按活跃域服务求和并受 rank 饱和值限制，容量只汇总活跃域；DRAM 保持 NUMA-rank 共享。旧 schema-v1 calibration 可读，未携带 placement 的 planner DAG 继续走 rank 聚合兼容路径，因此本变更不扩大 shape/width/window/ordering 搜索，也不宣称 production placement 已闭合。|
| 2026-08-14 | v1.19 | 增加显式 `calibrate_moe_planner_quick()` 部署校准：复用 schema-v2 的相同 service 定义和 builder，仅采 powers-of-two 到 16、LLC 域半宽/全宽与 rank 全宽，并降低 warmup/run；同构域复用代表性 LLC probe，异构域补 LLC-only probe。函数恢复调用者 affinity、Torch 线程数和 SVE dispatch 环境，原子输出并默认拒绝覆盖；不在 import/首请求运行，不做 operator residual fit，不改变 planner 候选、公式或默认 dispatch。|
| 2026-08-14 | v1.20 | 将显式 quick calibration 与 analytical planner 纳入可安装 Python 包，并增加线程安全的进程级 `MoePlannerRuntime` 注册。只有调用方显式安装且调用属于同一 ordered CPU rank、standalone/TP、SVE BF16 fused-SiLU、完整本地 expert 域时，normal `fused_moe_bf16_tiled` 才降低为现有 Plan V2；清空 runtime 或任何兼容性不匹配均保持旧 dispatcher。生产 runtime 首版用 analytical `T_iso` 对 homogeneous team 做 bounded LPT 搜索，只生成 strict Plan V2；完整 mixed-shape/phase-DAG/tail 搜索仍保留为离线 planner。Plan V2 schema 不变，native analytical scoring、性能精调及 EP 分片支持后续完成。|
| 2026-08-16 | v1.21 | 优化 production quick planner 的 homogeneous LPT 实现，不改变数学问题或候选：每个候选宽度只对 distinct route count 计算一次 analytical `T_iso`，用 `(load,lane)` heap 替代通用 mixed-width 逐 lane cost 扫描，并显式保留 `load + cost` 浮点舍入同分时的低 lane-id tie-break。m5 TP2、H4096/F1024/E256、43 层真实 DSV4 路由上，两 rank 的 materialized Plan V2 与旧实现逐字节一致；双 rank 并发、每层 7 次 forced-miss 的 planner-overhead layer-median 从 123.892/125.917 ms 降至 35.633/35.378 ms（逐层收益中位数 71.15%/71.87%），cache-hit 路径保持在约 32.6 ms。候选、目标、不确定性、early merge、schema、ABI 与默认 dispatch 均不变。|
| 2026-08-16 | v1.22 | 将 production analytical quick planner 的 homogeneous heap LPT、候选评分与最优 shape 选择等价迁移到单线程 C++；Python 继续计算精确 distinct-route `T_iso` cost rows，pybind 只完整转换胜出候选、其余返回 ranking 摘要，扩展不可用时回退 Python。m5 TP2、H4096/F1024/E256、43 层真实 DSV4 路由上，两 rank 的 Plan V2 与 v1.21 逐字节一致；双 rank 并发、每层 7 次 forced-miss 的 planner-overhead layer-median 为 31.998/31.739 ms，相对 v1.21 逐层收益中位数 10.15%/10.01%，累计相对原始 generic LPT 为 74.17%/74.73%。本阶段固定一个 planner worker，且不改变公式、候选、剪枝、排序、schema、ABI 或默认 dispatch。|
| 2026-08-16 | v1.23 | 为 analytical quick planner 增加固定 candidate-index 的候选级 OpenMP 并行，并复用 `FUSED_CPP_MOE_PLANNER_THREADS`/构造参数；每个候选内部仍单线程且按原索引归并，43 层双 rank Plan V2 在 1/2/4/8 workers 下保持逐字节一致。m5 TP2 双 rank sweep 中，2 workers 相对同二进制 1 worker 的 forced-miss planner 逐层收益中位数仅 0.19%/0.15%；4 workers 为 0.13%/-1.55%，8 workers 为 -0.26%/+0.80%，均未达到 10% 门槛。因此 production quick 未配置时继续使用 1 worker，多线程只保留为显式诊断能力。|
| 2026-08-16 | v1.24 | 将 analytical DAG active-phase pressure 计算改为固定 8-resource demand/time tuple，并缓存 immutable phase 的 isolated `base_ns`；scalar 诊断访问器、公式、归约顺序、event 边界和 early-merge 决策保持不变。m5 TP2、43 层真实 DSV4 路由、双 rank 并发、每层 7 次 forced-miss 中，planner-overhead layer-median 从同二进制 1T 对照的 31.823/31.676 ms 降至 23.565/23.549 ms（逐层收益中位数 26.00%/26.17%），累计相对原始 generic LPT 为 80.97%/81.33%；两个 rank 的 materialized Plan V2 仍逐字节一致。|
