# CPU MoE 调度数学模型

> Runtime integration note (updated 2026-08-21): machine calibration remains an
> explicit deployment action. An installed `MoePlannerRuntime` binds one
> analytical model and `PlannedMoE` quick planner to the calibration's ordered
> CPU rank. Compatible standalone/TP SVE fused-SiLU calls lower to Plan V2;
> calls outside that domain preserve the original dispatcher. Production quick
> search evaluates homogeneous-team isolated-LPT candidates in native C++ when
> available, disables route-plan caching by default, and can precompute a dense
> versioned `T_iso[M,T]` disk cache. `FUSED_CPP_MOE_PLANNER_FIXED_THREADS`
> selects one homogeneous fixed-width LPT fallback. Full analytical search
> evaluates mixed-width strict, temporal-order, tail-pool, and bounded-tail
> candidates with the phase DAG, but is an offline performance-oracle path: the
> current 80-core DSV4 run evaluated 469 candidates and took about 42 seconds
> cold. Plan V2 schema and numerical execution semantics are unchanged.

> 状态：调度问题定义的 source of truth。
>
> 最后更新：2026-08-21。论文 claim、证据和未闭合 gate 的统一索引见
> [`../docs/moe_paper_readiness.md`](../docs/moe_paper_readiness.md)。
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

N domain 先按 worker 划分：worker $j$ 独占一条连续 stripe

$$
\sigma_s(j)=\left\lfloor\frac{q_s}{t}\right\rfloor+\mathbb 1\!\left[j<q_s\bmod t\right],
$$

即前 $q_s\bmod t$ 个 worker 各多一个 tile（与 native `split_evenly` 一致），窗口在各自
stripe 内部切分（thread-major）。令每线程窗口为 $w_s$ 个 tile，则

$$
g_s=t w_s,\qquad
R_s=\left\lceil\frac{\max_j\sigma_s(j)}{w_s}\right\rceil
=\left\lceil\frac{\lceil q_s/t\rceil}{w_s}\right\rceil
=\left\lceil\frac{q_s}{g_s}\right\rceil,
$$

最后一步是取整恒等式，故 pass 数与团队每 pass 消耗的 $g_s$ 个 tile 都与旧的
window-major 映射相同；不同的只是 worker 所属的列。$w_s=\lceil q_s/t\rceil$ 即
full-stripe、$R_s=1$ 端点。

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

当前 production planner 仍只优化 2.4 节的 expert-compute makespan。选定 compute
plan 后固定写入 `early_merge=true`，不再为 merge policy 运行逐 task completion-time
DAG，也不读取 `topk_ids` 做 routing-tail/burst gate。该 policy 不增加 shape、width、
window 或 ordering 候选，不改变候选评分、剪枝和 histogram cache identity；它只是固定
Plan V2 的执行策略。

这个固定 policy 是实测 operational default，不是 active cost model 推导的最优性结论。
m5 TP2 DSV4 的 43 层真实路由上，旧 auto 在 43/43 层本来就由 native team-load
heuristic 解析为 on；强制 on 的 kernel 总时间与 auto 差异小于 0.06%，而跳过 planner
DAG 显著降低 cache-miss 开销。其他 workload、机器和非 direct-route backend 不在这组
性能证据的声明域内。

历史 TP4/F512 long/short bimodal 的 temporal-order 实验中，关闭 early merge
曾带来约 5.88%--6.07% 收益；全局固定 on 会放弃该保守 gate，因此这是已知的
cross-workload 回退风险，而不是尚未观测到的理论风险。

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

### 6.5 LLC-domain cold-phase shortlist master

离线 heavy planner 在 6.2 的 cold-phase CP-SAT 上做增量扩展，不把完整 nonlinear event
simulator 直接编码进整数规划。令 LLC domain 集合为 $\mathcal D$，domain $d$ 的连续 logical
core 区间为 $[b_d,b_d+C_d)$。第一版对 expert $i$、width mode $t$ 和能容纳该 width 的
domain $d$ 引入：

$$
y_{itd}\in\{0,1\},\qquad
\sum_{t,d:t\le C_d}y_{itd}=1.
$$

每个 $(i,t,d)$ 复用 6.2 的 optional master interval 和有序 W13/W2 cold/steady phase；
CPU cumulative constraint 从 rank-global 改为每个 domain 独立：

$$
\sum_{i,t:\tau\in[s_i,e_i)}t\,y_{itd}\le C_d,\qquad \forall d,\tau.
$$

cold packed-B 仍共享一个 NUMA-rank DRAM cumulative constraint，依赖边保持
$s_i\ge e_j$。因此 master 只搜索 whole-expert width、LLC domain 和 start/order；不搜索
stage window、W13/W2 独立 width、route slice、tail pool 或 merge policy。给定
$(M_i,t_i)$ 后，W13/W2 window 继续由 9.36 的确定性 policy 生成。

**Incumbent。** 当前 analytical full plan 可能包含跨 LLC 边界的 lane，不能直接成为
single-domain master 的可行解。实现保留其 per-expert width 和 lane order 信息，先用同一个
cold-phase master 只修复 domain/start，得到 projected feasible incumbent；projection 与
mixed solve 都计入 $T_{plan}$。该 incumbent 同时作为 CP-SAT hint 和显式 upper bound。即使
求解器在时限内只有 lower bound $L_{CP}$，仍可报告：

$$
gap_{CP}=\frac{U_{inc}-L_{CP}}{U_{inc}}.
$$

该 gap 只证明 fixed-rate cold-phase surrogate 的近似最优性，不证明硬件 makespan 的
近似最优性。6.4 的 resource bound $L_R$ 另行报告；若容量来自实测饱和服务而非架构上界，
必须标为 calibrated-model relaxation，不能称 hardware certificate。

**Strict greedy union。** 第一阶段不启用 tail pool、tail repartition、stealing 或 resize。
运行时 quick planner 产生的 homogeneous LPT greedy plan $G$ 保留原始 per-expert width、
`core_begin`、lane order 和 dependency，不投影到 single-domain plan。固定 greedy 分支仍用
cold-phase CP-SAT 优化合法 phase start/wait，但 overlapping core interval 必须由 dependency
path 排序。domain-aware SAT 分支的可行域记为 $\mathcal F_D$，第一阶段完整域为：

$$
\mathcal F_{strict}=\{G\}\cup\mathcal F_D.
$$

$\mathcal F_D$ 的 width domain 仅由 runner 显式声明的已校准宽度与 exact greedy 使用的宽度
组成；full/one-step plan 只提供 hint，不得把其余宽度隐式扩入 proof master。若 hint 含域外宽度，
改用 exact greedy plan 作为可行 hint。普通 analytical full controls 仍可使用 machine profile 的
其它 supported widths，但它们不属于本次 CP gap 所证明的集合。

两个分支不允许逐 task 混合，因此可分别求解再取最小，严格等价于一个 plan-level disjunction：

$$
U_{strict}=\min(U_G,U_D),\qquad
L_{strict}=\min(L_G,L_D),\qquad
gap_{strict}=\frac{U_{strict}-L_{strict}}{U_{strict}}.
$$

故 $U_{strict}\le U_G$ 在相同 cold surrogate 下成立。domain solve 的 hint 只影响搜索效率，
不定义 $\mathcal F_{strict}$；实验中可继续使用 one-step strict projection 作为 domain hint，
同时 exact greedy 始终由独立分支保留。该保证不自动推出硬件 wall time 不劣，必须用相同
strict executor、输入、权重轮换和 paired runs 验证。动态尾池属于后续在线 recourse 策略，
不由本节 union gap 覆盖。

最终 executable event union 还必须显式保留当前 analytical full incumbent、one-step
incumbent 与 exact greedy，不允许 CP solve 后把这些已知可执行计划丢掉。CP shortlist 与三种
incumbent 一起由完整 event model 排序：

$$
P^*_{event}=\arg\min_{P\in\{P_{full},P_{one},G\}\cup\mathcal S_{CP}}
\widehat T_{event}(P).
$$

因此“proof gap”与“最终 plan”是两个不同结论：前者只证明 fluid cold surrogate 上的
$\{G\}\cup\mathcal F_D$，后者是包含 incumbents 的 measured shortlist decision；二者都必须
单独报告，不能把 incumbent 的收益记为 CP-SAT 新发现。

**Canonical executable search state。** Event-guided VND/LNS 在进入搜索前使用一个内部
version-1 fixed-lane state，而不是直接修改Plan V2数组。令rank logical cores为
$[0,C)$，lane集合为：

$$
\mathcal L=\{(b_l,t_l,Q_l)\},\qquad
b_0=0,\quad b_{l+1}=b_l+t_l,\quad \sum_l t_l=C,
$$

其中$Q_l$是lane内有序whole-expert task序列。每个active expert恰好出现一次；第一个task
无依赖，后续task只依赖同lane前驱。因此core interval和dependency由state唯一决定，每个
state天然是strict fixed executable DAG。独立的LLC-domain interval集合也必须按logical CPU
顺序无缝覆盖$[0,C)$；现有incumbent中跨domain的lane保持原interval，并记录所有相交domain，
v0转换不得为了拓扑整齐而改写基线。

state另外保留ordered physical CPU ids、per-task W13/W2 window tiles和plan-level
early-merge三态。canonical hash覆盖state schema、CPU/domain topology、lane partition、task
顺序、route counts、windows和early merge。当前适配器只接受strict、fixed、unsliced、
whole-expert lane chains；tail pool、route slice、resize和一般interval DAG显式拒绝。planner
result $\rightarrow$ state $\rightarrow$ planner tasks/Plan V2 bridge必须逐字段往返，且同一
analytical event scorer结果不变。该内部状态不修改public Plan V2 schema、runtime ABI、
planner candidate space或默认dispatch；它只是后续executable-neighborhood search的合法性
边界。

**Step-1 executable order neighborhood audit。** 在不改变lane interval、team width、window
或early-merge语义时，首个局部搜索域只允许五类whole-expert move：同lane相邻交换、同lane
插入、同宽lane间迁移、同宽lane间交换，以及同宽跨LLC-domain迁移。每次move直接构造上述
canonical state，依赖重新由lane序列派生；因此不存在先搜索fluid解再lowering的误差。

placement-aware event trace为task $i$ 定义诊断criticality：

$$
q_i=\sum_{e:i\in A_e}\Delta_e d_{e,i}
\left(1+\frac{s_e+\Delta_e}{T_{event}}\right),
$$

其中$A_e$、$s_e$和$\Delta_e$分别是event的active tasks、起点和持续时间，$d_{e,i}$是该
task在event内的phase dilation。该式只用于选择要移动的expert，不改变objective，也不作为
新的cost-model参数。审计以相同expert数、相同per-operator评分预算比较$q_i$最高集合和
uniform-random集合，分别记录proposal、valid、duplicate、unique、event improvement及评分
吞吐；两组event top candidates再以同一输入/权重、随机交错paired rounds实机执行。只有存在
稳定实测改进、critical引导明显富集优质候选、且event改进没有被硬件系统性反转时，才进入
VND。否则该结果用于拒绝当前order-only LNS结构，而不是继续增加自适应权重。

**Width-specific expert overhead。** Temporal counterexample显示，当1T lane串行执行许多
whole experts时，只用GEMM/cache service会低估其绝对lane完成时间，即使active-set slowdown
的相对趋势基本正确。将已有global runtime overhead扩展为离散width override：

$$
O_{expert}(R,t)=a_t+b_tR,
$$

其中未校准$t$严格回退global $(a,b)$，width之间不插值或外推。该项作为每个expert的串行
operator phase加入`predict_expert`，不乘LLC/DRAM或wide-team dilation。首个Arm 80C probe
固定同一1T victim lane和任务序列，比较all peers delayed、仅4x16T peers concurrent、再加入
15x1T peers concurrent三态，并以31轮随机交错trace中all-delayed的per-task median非负线性
回归拟合$(a_1,b_1)$。planner真实trace及其neighborhood候选不参与拟合，只作为后续holdout。

**Uncertainty-aware executable decision。** Point event score仍可能把低于模型分辨率的
temporal delta排成确定顺序。对order-neighborhood候选增加不冒充统计上界的lane guard：

$$
T_{guard}(P)=\max\left(\widehat T_{event}(P),
(1+\epsilon)\max_l\sum_{i\in Q_l}T_{iso}(R_i,t_l)\right).
$$

$\epsilon$读取machine calibration的systematic relative uncertainty。第二项用于阻止新重lane
被point event错误隐藏在旧critical path之下，但不声称是硬件时间上界。候选只有满足
$T_{guard}(P')\le(1-\delta_{min})T_{guard}(P)$才可自动替换incumbent；首版
$\delta_{min}=2\%$与既有no-regression/actionability gate一致。低于margin的候选可进入
measured shortlist，但不能驱动VND接受或作为模型证明收益。

**离线双候选选择（Lab，2026-09-06）。** 对现有 strict shape/order 候选集及 analytic
baseline，保留 $P_m=\arg\min_P\widehat T(P)$（沿用现有 tie-break）和历史 width-fallback
选择 $P_f$。以完整 PlanV2 bridge 的确定性 JSON 内容去重：
$S=\operatorname{unique}_{bridge}\{P_m,P_f\}$，因此 $1\le|S|\le2$。
该去重不重排 task，也不宣称能识别所有语义等价的不同编码。单候选直接保存，标记未进行
硬件比较；双候选沿用两次独立进程、每次5 warmup与31 paired rounds、同copy配对、4-copy
轮换和216 MiB scrub。每个session取中位耗时最小的计划，只有两个session一致才输出
`hardware_consensus_winner`及完整bridge；不一致输出未决，不自动回退并冒称硬件winner。
若两次session对另一计划的paired gain中位数均大于2%且P10大于0，才标记`actionable`。
较小但一致的经验winner可保存，不能当作稳定收益保证。winner携带route/layer、calibration、
extension、计划hash及session证据；证据不匹配或轮次不完整时拒绝生成。
这只改变独立离线Lab工作流，不改变模型公式、生产selector、partial-order pruning或PlanV2
格式。已知三trace中median/high-skew支持score-best，uniformish完整bridge相同；不是新的
未见数据泛化验证，更不是全局最优证明。入口为`bench_selection_pair.py tune`，历史session
可通过`offline_selection_winner.py`直接生成结果，避免重复硬件测量。

**有预算的单层order-only扩展（Lab）。** 以已保存硬件winner为anchor，保持每条lane的
CPU区间、width、task集合、routes、window和early-merge不变，仅调整序列并重新派生链依赖。
三类proposal为随机子集lane独立反转、随机非零循环位移、多lane联合交换head/tail；每类
最多256次尝试得到24个唯一候选。保留每类event-score最小者及与其task位置Hamming距离
最大者，跨类别去重并保留anchor；每trace最多72次候选评分、7个硬件计划，只有一层。
这不是全排列搜索，也不是新的默认neighborhood。两次session均满足paired gain中位数>2%
且P10>0才考虑后续层；不根据结果追加预算。实现及证据见
`bench_bounded_order_extension.py`和对应20260906报告。

**访存压力平滑proposal（Lab，不是时间模型）。** 固定每lane task集合和width，将isolated
phase串接，令每phase offered rate为其既有isolated DRAM字节估计除以base时长。
以$q(t)$表示各lane rate之和，候选目标为$J=\int q(t)^2dt/T$；固定工作与isolated horizon
保证其变化不是总流量变化或主动延迟。domain版本使用$\int\sum_d q_d(t)^2dt/T$，按lane
CPU落入LLC domain的比例拆分offered rate；不假设两domain有独立DDRC容量。它只是生成
顺序候选的近似指标，忽略真实contention引起的phase位移，不能解释为实测带宽/queue。
每目标64次mutation，生成global平滑、domain平滑、global增峰反向对照，加routes大小交替
策略和anchor，每trace最多5个硬件计划。两session硬件gate沿用中位gain>2%、P10>0。
不改v8公式、calibration或生产默认，不能根据本轮结果反向拟合物理项。

GEMM平均密度对照（2026-09-07）将每个W13/W2的连续phase分别合并成单个区间，
$\rho_{gemm}=\sum_p B_p/\sum_p T_p$，非GEMM区间保持独立；总流量和时间守恒。
只生成global二阶矩下降/上升各64次尝试的终点，加anchor共3计划。主要比较均匀计划
相对非均匀计划的实测paired gain，另行检查相对anchor是否满足原actionable gate。
此Lab模式不修改原phase模式、不代表真实访存速率、也不进入生产评分或pruning。

合成M验证（2026-09-07，Lab）固定E60、10x8T、每lane三小三大任务，比较全部LLLHHH
与奇数lane HHHLLL，关闭early merge；全同M参考为1/4/16/64/256。两session中(4,32)、
(16,128)、(64,512)交错收益约3.4%、25.2%、17.4%，(1,8)无稳定收益。平均DDR流量
随大M下降，但收益非单调，因此不能用平均压力或M阈值直接替代实测选择。此结果只属合成
固定宽度验证，不新增生产规则、物理参数或真实trace泛化结论。

显式交错模板与扩展顺序（2026-09-07，opt-in Lab）：`interleave`生成按routes升降序的
lane奇偶错排（两种方向），以及lane内大小任务按1/2个为块交替（两种起始方向）；同M
lane保持不变。`extended`再加入独立反转、循环位移、多lane head/tail交换、完整连续块
迁移，每随机family最多256次尝试、24唯一候选。每个操作保留每lane task集合及width、
CPU/window/merge语义，依赖由序列重新派生。新模式先为每family保留一个代表，再用剩余
名额；模板/反转/head-tail用模型最好代表，rotation/block用最大顺序距离代表，最多6候选
加anchor。分类标签和moved experts进入artifact。默认`legacy`不变；此接口不直接扩展
生产neighbor或VND/LNS自动接受域，也不复用旧residual radius作新family的确定性剪枝。

策略对照（Lab，2026-09-07）：interleave/extended各自最多6候选加同一最新已测anchor；
跨策略按完整bridge去重，显式union测量上限13，默认单策略上限7保持不变。共享候选只测
一次，归属映射保留；不足预算不补样，报告生成/评分数与实际测量数。新的anchor输入必须
通过原始frontier对应双session的稳定收益gate，并以原始routes metadata完整恢复bridge。
这不是生产接受规则，不把已选union当成全局oracle，也不因新策略标签复用旧残差剪枝。

**Anchor-relative partial-order diagnostic。** 离线搜索不需要把低于分辨率的候选强制排成
全序。固定一个可执行anchor $P_0$，采用“正值表示candidate更快”的相对增益：

$$
\widehat g(P;P_0)=100\left(\frac{\widehat T(P_0)}{\widehat T(P)}-1\right),
\qquad
g^{hw}(P;P_0)=\operatorname{median}_r
100\left(\frac{T_r(P_0)}{T_r(P)}-1\right).
$$

校准样本的delta residual为$e=g^{hw}-\widehat g$。首版不修改event mean，只取
$|e|$的预声明coverage分位数$q_c$；context样本不足时依次回退operator family与全局
分位数。对未进入生产的离线诊断关系：

$$
P\prec P_0\iff \widehat g-q_c>\delta_d,
\qquad
P_0\prec P\iff \widehat g+q_c<-\delta_d,
$$

其中$\delta_d$是dominance margin，与硬件actionable threshold分离；区间覆盖anchor的
pair保持incomparable。首版只比较candidate与当前可执行anchor，避免任意pairwise classifier
形成偏序环。

**Offline shortlist/search policy。** 离线runner只加载带replay gate和匹配
calibration/extension identity的pairwise report。候选分为：`better`进入acceptance集合，
`incomparable`保留在硬件frontier，只有`worse`可由partial order做dominance pruning。
有限budget导致的未入选单独记为`budget_deferred`，不得记作模型pruning。shortlist先按context
各保留一个leader，再按区间下界和point gain补满；anchor始终隐式保留。搜索每轮只接受
完整下界超过actionable margin的best-improvement，并以该状态成为下一轮anchor；没有
`better`即停止，不能把`incomparable`当作下降失败证明。启用前必须满足对应top-K的
measured-best recall与zero-false-pruning gate；首版默认$K=16$。该策略只进入离线Lab路径，
不改变production quick planner、Plan V2或runtime dispatch。

**独立 context-aware residual 候选（Lab）。** 不修改冻结 v8 或现有搜索 objective，
实验模块 `optimizations/fused_moe_sve/benchmarks/context_aware_residual.py` 预测

$$
\widehat T_{ctx}(P)=\widehat T_{v8,event}(P)+10^6(b+\beta^T z(P)),
\qquad z_j(P)=(x_j(P)-\mu_j)/s_j.
$$

时间输入/输出为 ns，拟合 residual 为 ms；$\mu,s$ 只用训练计划，常量列令 $s=1$。
训练目标为 $n^{-1}\|y-b-Z\beta\|_2^2+\alpha\|\beta\|_2^2$，
$y=(T_{hw}-\widehat T_{v8,event})/10^6$，截距不惩罚，首轮固定 $\alpha=10$。
输出非正或非有限时间直接报错，不静默裁剪。此为经验 residual，不声称新增物理定律。

首版仅支持显式 LLC domain、非空 strict whole-expert plan、W13/W2 `window_tiles=0`。
23 个白名单特征按结构（8）、预测时序（12）、交互（3）递增做并列消融：lane isolated
load 与 domain core-time、head/tail route fraction；每 domain 的 B-active task count、
requester count 与时间加权波动、small-route exposure、operator/B overlap、cohort
transition、lane finish spread；count/requester 与 small-route exposure 交互。
small-route 指 $M\le12$，不表示权重更小；B-active 是预测的 cold_b/steady_b phase，
不是实测 DDR stream，也不等同 transfer-bound。跨 domain task 在每个触及 domain 计一次，
requester 按实际 active CPU 分摊；统计按 domain 等权及事件时长加权。

特征从完整 plan 与重新生成的冻结 v8 预测事件提取，不使用实测 PMU、实测结束时间、
expert/hash identity 或 search outcome。时间线仍是 v8 近似；本版 plan-level residual
不反馈到 phase simulator，不声称解决队列动态或阶段误差归属。model profile 独立序列化，
校验 v8 calibration SHA、extension SHA、重评分源码 SHA、shape 和 protocol identity。
特征语义变更必须更新独立 schema，不能借用 production calibration schema。

每个训练 plan 仅一条聚合 observation。验证默认拒绝与训练的 canonical plan hash 或
structural group 重叠；显式 replay overlap 只作诊断。同 plan 的另一 session 不是新 plan
holdout。报告 MAPE/MAE、同 group 同 session 内 pairwise gain error 与 2% median-margin
direction、top-K measured-best recall/regret、训练范围外特征；median-margin 不冒充 paired
round 的稳定置信标签。$K$ 覆盖整个已测组时标记 trivial recall。无剪枝政策，false pruning
为 not applicable 而不是 0；`Accept=Prune=empty`，不得复用旧 partial-order radius。
保留旧 route sweep/三 trace holdout；后续已公开 LNS 数据仅用于 development，不能因重放
收益而宣称独立硬件验证或采用。运行实验由用户指定的 terra/medium agent 执行。

**Two-level executable evaluator。** 完整placed event score保持唯一权威objective，并按
canonical state hash缓存。cheap screen复用不可变lane phase及isolated lane load，构造
lane bound $L$、按core-time平均的rank bound $R$、按lane与LLC domain物理交集分摊的
domain bound $D$，再以忽略完整resource vector、仅保留wide/narrow team residual的
phase surrogate $S$排序：

$$
B(P)=\max(L(P),D(P),R(P)),\qquad
S_{priority}(P)=\max(S(P),B(P)).
$$

$S_{priority}$不是完整event objective或硬件上界，只能筛选进入完整评分的diverse subset。
如果任一声明budget丢失audit corpus的measured best，则不得启用该screen；增加吞吐本身不
构成采用证据。

**Measured-shortlist context contract。** 对进入硬件shortlist的真实候选，离线artifact必须
保存相对anchor的真实affected lane集合。每条affected lane记录物理CPU/LLC、isolated load
以及完整有序$(expert,routes,window)$序列；placed event log压缩为affected-task的head/tail
暴露、solo time、phase/team dilation、peer threads、resource overlap、cohort transition及
critical lane/task摘要。原始event序列不进入artifact，以控制体积。pairwise residual context
可以使用这些解析模型产生的标签，但不得使用硬件结果本身构造标签。

high-skew诊断中，三个大幅退化的same-width cross-lane swap都把`62/68`-route的1T lane尾
任务换到另一1T lane头部，使affected-tail暴露增加`4.02--5.70 ms`；硬件退化
`12.61--15.17%`。无新增tail暴露的swap硬件median为`+0.40%`且区间跨零；四者均没有
placed critical-lane/expert switch。因此该批反例首先按tail exposure而非critical switch分组。
但独立`68-route tail <-> 1-route head` probe的background/isolated硬件与模型增益分别为
`-32.40/-34.35%`和`-34.97/-34.80%`，只留下`+1.95/-0.17`个百分点residual。该物理转换
已由现有lane/event模型表达，未复现real-trace剩余residual，故不得增加新参数或修改冻结
calibration；需要找到更具体且能在独立probe上复现的context。

**Topology-preserving width neighborhood。** Step 3仍在canonical executable state内搜索，
不允许先生成fluid assignment再做contiguous lowering。设lane $l=(b,t,Q)$ 完全包含于单一
LLC domain $d$。若$t/2$属于已校准宽度集合，则split将其替换为
$(b,t/2,Q_0),(b+t/2,t/2,Q_1)$；若两个物理相邻、同域、等宽lane
$(b,t,Q_0),(b+t,t,Q_1)$满足$2t$已校准，则merge将其替换为
$(b,2t,Q')$。$Q_0,Q_1,Q'$只由受影响lane中的expert组成，并用新宽度的isolated-time LPT
确定性重排。既有跨domain lane可保留，但禁止split、merge或作为width migration端点，因而
LLC-domain partition和所有未受影响lane保持逐字节不变。

第三类move在同一domain内的现有lane之间迁移一个whole expert，并要求两个lane宽度在有序
已校准集合中相邻。源、目标lane原有task的相对顺序不变，目标插入点被显式枚举；被迁移task
在执行时采用目标lane宽度。任一task从$t$变为$t'$时，必须通过已有确定性policy重新计算
$(w_{13}(M,t'),w_2(M,t'))$，不得沿用旧宽度窗口。三类move均保持
$\sum_l t_l=C$、每个active expert恰好一次、strict fixed whole-expert依赖和Plan V2 v2
执行语义。order-only、width-only及二者候选并集使用相同per-operator预算、canonical hash
去重、完整placed event scorer、lane guard与2\% actionable gate；该审计不改变production
planner候选或默认dispatch。

**Lane-atomic template-level LNS。** 局部beam在depth 3停止后，离线Lab允许一次同时改变
多个lane宽度、domain placement与lane内顺序的destroy/repair，但仍直接生成canonical
executable state。对critical expert $i$所在lane $c(i)$和目标destroy数
$d\in\{4,8,16\}$，分别在domain-local与cross-domain两个scope内选择包含$c(i)$的连续
lane块：

$$
B^*(i,d,s)=\arg\min_B
\left(|Q_B|-d,\ |B|,\ \sum_{l\in B}t_l,\ \operatorname{first}(B)\right),
\qquad |Q_B|\ge d.
$$

domain-local要求$B$中每条lane完整属于同一LLC domain；cross-domain要求$B$触及至少两个
domain，且repair产生的每条新lane都完整落在某一个domain内，不允许引入新的跨domain lane。
$d$只是目标数；由于lane是拓扑原子，实际destroy集合是$Q_{B^*}$，artifact必须同时保存
operator中的target $d$和完整`moved_experts` closure，不能把$d=4$解释成只移动4个expert。
块外lane对象、CPU顺序、task/window与early-merge逐字节不变。

对块的core span $C_B$，宽度模板集合为：

$$
\mathcal W_B=\left\{(w_1,\ldots,w_m):
w_j\in\mathcal T_{cal},\ \sum_jw_j=C_B,\
1\le m\le\min\left(|Q_B|,\max(|B|+4,8)\right)\right\}.
$$

首版每个块最多保留4个预声明模板：一半按相对原width histogram与lane count的距离最近，
一半最远，从而同时覆盖小改动和CP-SAT-like异构跳跃。repair按
$\max_{w\in W}T_{iso}(R_i,w)$递减处理expert；placement beam以
$(\max_lL_l,\sum_lL_l^2,\max_lL_l-\min_lL_l)$排序，并强制每条新lane非空。
$d=4/8/16$分别使用beam width $16/32/64$。每个placement再生成load-desc、reverse、
critical-head与critical-tail四种确定性顺序，并为变宽task重新计算W13/W2 window。实现可以对
同一$(i,w)$预计算window retarget，并用增量expert-id signature做last-write-wins去重；排序键
与四种顺序策略保持不变。这里没有新增cost-model参数；$T_{iso}$只指导repair，完整placed-event
score仍是shortlist point estimate。

正式high-skew首层从full、`4115ab99...`与`514ced0d...`三个实测incumbent各做两个独立
proposal restart；critical与random各按六个scope/size operator抽50个，partial order只做
one-sided worse pruning并保留top-16。模型共评估3,500个unique candidate、3,512次event
call，用时1,193.47 s；对照已关闭local beam为3,535次event call、507.06 s。两者硬件预算
均为106个plan。LNS两次独立31-round session得到42个跨session stable comparison、40个
unique winner和0/11 false-pruning sentinel；全部stable winner都来自cross-domain repair。
共识state `189d70b0...`为`16T,16T,8T,8T,8T,8T,...`，相对`4115ab99...`的paired
median为`+3.880/+3.271%`、P10为`+1.974/+1.506%`，绝对median为
`31.249/31.297 ms`。两个session的absolute best不同，但共识winner在第二场仅落后
`0.0076%`。因此在high-skew上接受template-level LNS结构并保留硬件rerank；该结论尚未覆盖
median/uniformish，不允许进入production或声称三trace最优。

**Template-LNS cross-trace decision。** high-skew共识elite再扩一层时，从`189d70...`与
`a7b382...`各做两个restart，得到2,343个unique candidate、2,351次event call和858.70 s
search wall。72-plan双session只留下一个严格stable的domain-local d8 comparison；归一化
absolute best `37bac...`为`30.406/30.557 ms`，相对`189d70...`的absolute-median比值收益为
`3.518/2.129%`，但第二场paired median只有`1.873%`且P10为负。因此保留该elite记录，停止
high-skew depth-3自动扩展。

随后保持相同`local/cross x d4/d8/d16` operator mixture、beam `16/32/64`、4 templates/block、
每operator 50、每parent两个restart，在median与uniformish上从各自full/one-step/greedy/
fixed-width executable control开始。control在目标Arm进程内由冻结route+v8重建并立刻保存
canonical state/PlanV2；median四个hash与旧VND完全一致，uniformish的full/one-step/
fixed-width按hash合并为一个parent。

三条trace的硬件共识winner都在两场相对各自strongest preserved anchor超过2%：

| trace | winner | median ms, session 1/2 | gain vs strongest anchor, session 1/2 |
| --- | --- | --- | --- |
| high-skew | `189d70...` | `31.249/31.297` | `3.961/3.428%` |
| median | `2ab435...` | `31.381/31.393` | `2.671/2.863%` |
| uniformish | `50a7d4...` | `31.517/31.698` | `2.057/3.250%` |

因此接受fixed template-LNS neighborhood作为离线硬件辅助搜索域。但原partial-order radius
由local-move residual校准，不能外推到large closure。median的两条`candidate_worse`
sentinel在两场都稳定为正：absolute best `2ab435...`被模型预测`-3.770%`且upper bound
`-2.001%`，硬件paired median为`+2.290/+3.102%`；`c8487e...`预测`-3.842%`，硬件为
`+3.870/+3.934%`。uniformish的26个model-better中25个通过两场strict stable gate，另一个
虽median为`+3.054/+3.556%`，但至少一场P10未过零。故LNS的最终决策集合必须满足：

$$
\operatorname{Prune}_{LNS}=\varnothing,\qquad
\operatorname{Accept}_{LNS}=\varnothing.
$$

partial-order relation仍序列化为diagnostic，但所有未测候选只能记作`budget_deferred`，不能记
dominance-pruned；`candidate_better`不能更新incumbent。有限硬件budget不再按relation或
残差半径分配。template-LNS使用relation-agnostic categorical farthest-first selector
`relation_agnostic_categorical_farthest_first_v1`：按canonical hash去重后，强制覆盖可用
operator、score quantile（含0与4）、actual closure bin、width histogram与LLC-domain
assignment，再用无学习权重的farthest-first填满每start的top-16；top-16必须是top-32
audit的前缀。硬件frontier角色为`lns_diverse_top16`、`lns_diverse_audit_top32`和
`anchor`。冻结measured-suite设计回放在K=16上四条case的绝对实测最好、共识winner与
selected-best regret均为零。独立median holdout（seed `20261010`，每parent一次restart）
的nested-recall gate通过：top-16在两场都含top-32绝对最好与共识winner，regret为0，
K=8前缀已经零regret。该seed相对strongest full的绝对median收益为`+2.579/+1.916%`，
第二场低于2%，记作neighborhood/proposal失败而非selector漏召回。因此采用selector v1
作为离线LNS硬件shortlist，不改默认K=16，不把该seed当作production或全局最优。该安全
默认只影响template-LNS Lab runner；local VND的已校准family comparator语义不变，
production planner、Plan V2、runtime与v8 mean全部不变。

独立median之后，Lab runner补齐search-cost split：closure、width template、beam repair、
assemble、canonical hash/去重、screen、exact event，以及shortlist（diagnostic partial-order
labeling加上feature/quantile构造）。冻结seed `20261010`的生产路径在等价beam rewrite后为
687.55 s：exact 341.43 s、shortlist 278.77 s、sample 56.31 s（其中beam 40.56 s）。隔离profiler
（`window_selector=(0,0)`、单strategy）曾把enumeration热点定在beam（full 7.045→2.245 s，
one-step 67.573→17.221 s）；生产路径上墙钟从797.01 s降到687.55 s，候选hash、抽样计数、
模型分数、quantile与有序top-16/top-32保持不变。event simulator未改；跨lane增量回放不能事先
当成等价。relation-agnostic selector本身不是279 s：全局farthest-first merge仅53 µs。
完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_lns_search_breakdown_beam_equiv_20260905.md`。

在冻结selector v1、K=16（按unique parent池化restart）和现有operator mixture下，把同一
2,400 exact-eval cap与同一85-plan硬件名额分给1 restart（N=50、4 start）和2 restart
（N=25、8 start）。两边都选出`0418b884...`为K=16硬件最好，且两场相对strongest full
都超过2%。2-restart的K=16集合与1-restart只重叠15/64；selected-best在两场都快于已知
median elite `2ab43572...`（`+0.797/+0.449%`），1-restart第二场相对elite为负。top-32外
16个分层样本没有打过K=16。2-restart搜索墙钟558.71 s对683.67 s。因此离线median LNS在
该预算下采用每parent两次restart、N=25，不改K、不改selector。这不是新的neighborhood
winner，也不是production结果。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_lns_restart_budget_20260905.md`。

同一冻结协议下把proposal seed换成`20261011`（stratified seed仍为`20261013`）后，
exact cap仍为2,400，LNS硬件名额仍为64 top-16加16层外抽查。搜索2,378次event、
2,362个unique、550.90 s，但没有生成上一seed的selected-best `0418b884...`；K=16
与`20261010`只重叠3/64。两场K=16最快解不同（`2fd99a11...` / `8deba718...`），
相对full为`−0.018% / +0.371%`，未过2%。注入的`0418b884...`在session 1仍是实测
最快（`31.122 ms`，相对full `+3.302%`），session 2与elite `2ab43572...`相差
`0.030 ms`。这是neighborhood/proposal失败，不是selector或2-restart分配失败。
离线median仍用2 restart + `N=25`，不改K、不改selector、不把`20261011`当作替代
proposal。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_lns_second_proposal_seed_20260905.md`。

对冻结median协议下相对reconstructed full绝对median至少2%的五个实测快plan
（`0418b884...`、`2ab43572...`、`7cac2afd...`、`1faf090a...`、`07355dde...`），
用shipped `enumerate_template_lns_neighbors` / `_sample_neighborhood` 在full parent
`98a32da5...` 的 `lns_00_r00`/`lns_00_r01` 上回放。五者都在该parent可达邻域内。
`0418b884...` 在seed `20261010` 被选中，在 `20261011` 于N=25 shuffle截断丢失
（index 373/452）；`7cac2afd...` / `1faf090a...` 在 `20261010` 的shuffle index为
40/716与26/1016，N=50会留下、N=25不会；`2ab43572...` 两边都进入event scoring但
parent-pooled rank在top32之外。Lab候选presampler
`structural_coverage_then_random_v1`（SHA256 `275dd663...`）与baseline共用同一次
shuffle、把每个非空closure bin的首次代表提前，design replay不改变这五个hash的
sampled membership，也不接入 `sample_template_lns_neighborhood`。不打开独立硬件
frontier。后续候选 `structural_coverage_closure_width_v1`（`e5a35b06...`）把coverage
key扩成`(closure_bin, width_histogram)`，同一回放在seed `20261010` 把已抽中的
`0418b884...` 挤出N=25，且未恢复任何sampling-lost tracked hash，判定reject。随后冻结
full-parent critical cross-domain d4 参考池：全局去重后该operator unique 452
（digest `a10eeba6...`），四个critical输入集合相同；历史session只测到11/452。
完整池诊断预算456 unique（去重后），约541 s/场，超出既有80 LNS-candidate槽，
不改默认N=25/K=16，Task 6B未授权。完整记录见
`optimizations/fused_moe_sve/results/template_lns_cursor_todo.md`。

**Narrow-team full-cohort correction。** 独立`2x1T concurrent -> 1x2T serial`
probe表明，现有shared-resource event方程对2T task在满载mixed background下的dilation
系统性高估，而不是漏掉1T slowdown。先用无背景的merged 2T逐task phase span拟合离散
operator overhead：

$$
O_{expert}(R,2)=a_2+b_2R.
$$

然后令$q_i$为前述team peer occupancy fraction，并对已校准narrow width $t\in\{1,2\}$
引入full-cohort residual correction $c_t>0$：

$$
k_t(q_i)=1+(c_t-1)q_i,
$$

$$
D_i^{final}=\max\left(1,D_i^{resource}D_i^{wide}k_t(q_i)\right).
$$

$c_t<1$表示修正解析resource dilation的系统性高估，不表示真实执行相对isolated获得加速；
最终dilation仍以1为下界。未校准width严格使用$k_t=1$，不按width插值或外推。首个Arm
80C校准固定相同两个target cores、4x16T加14x1T background、early merge off和三组与真实
planner trace隔离的合成短任务序列；$a_2,b_2$只使用`merge_isolated`逐task span，$c_1,c_2$
分别通过完整placed event simulator令三组`pair_background`和`merge_background`的median
log span error为零。isolated 1T pair和三条真实route trace不参与correction拟合。

**Placement-aware LLC event state。** 旧 analytical DAG 在 `_score()` 中删除
`core_begin`，所有 active task 共用 rank-global LLC working set、capacity 和 service；这会
把“task 分散在两个 LLC domain”与“task 全部挤在一个 domain”错误视为同一状态。当前
additive placed path 将 task 的 logical core interval 通过 planner `cpu_ids` 映射为真实 CPU
集合 $P_i$，再由 machine calibration 映射到 LLC domain。旧 `dag_makespan()` 保持不变；
有 topology 且调用方提供 placement 时使用 `dag_makespan_placed()`，无 topology 时回退旧
rank aggregate。

对 active phase $i$，令 $n_{id}$ 为其 active owner CPUs 中属于 domain $d$ 的数量，
$p_{id}=n_{id}/n_i$。第一版按 owner-thread 比例分摊 working set 和 LLC demand：

$$
W_d=\sum_i p_{id}W_i,
\qquad
q^{LLC}_{id}=p_{id}q^{LLC}_i.
$$

每个 domain 独立使用容量 $C_d$ 计算 miss fraction：

$$
m_d=f(W_d,C_d),
\qquad
m_i=\sum_d p_{id}m_d.
$$

task-specific $m_i$ 只作用于该 task 的 spillable DRAM demand；compulsory packed-B 首读保持
不变，DRAM contention 继续按 NUMA-rank 全局计算。LLC injection 在每个 domain 上分别得到：

$$
\rho_d=
\frac{\sum_i r^{LLC}_{id}}
     {B_d(\sum_i n_{id})}.
$$

多个 domain 还共享校准的 rank-level LLC fabric ceiling，得到 $\rho_R$。一个跨 domain gang
phase 由最慢 owner stripe 限制，因此其 LLC dilation 为：

$$
D_i^{LLC}=\max\left(1,\rho_R,\max_{d:n_{id}>0}\rho_d\right).
$$

当前 exact-N ownership 下 active owners 取 team CPU interval 的前 `phase.active_threads` 个
logical workers。输入校验要求每个 task placement 内 CPU 唯一、属于 calibration rank，并且
共享物理 CPU 的两个 task 必须由 dependency path 排序。placed event diagnostics 记录每个
domain 的 active threads、working set、spill、offered rate、capacity、utilization 和 dilation。
该修改只作用于 analytical heavy event scoring；empirical model、production quick planner、
Plan V2、runtime 和 kernel 不变。

**Wide-team concurrent pressure。** Placement-aware LLC/DRAM allocation仍无法区分
“相同 active cores、不同 fixed-team width”的同步与负载不均衡开销。例如 80C 上
$10\times8T$ 与 $5\times16T$ 具有相同总 core 数，但后者每个 expert 的 gang barrier、
owner-stripe 尾部和 phase 交接更宽。该残差不能从 isolated $T_{iso}$ 或 aggregate
bandwidth curve 单独识别，因此使用独立的 fixed-cohort control 校准，而不把它混入 LLC
service curve。进一步的 single-team control 证明超宽 gang 即使无 peer 也有内部 barrier/
owner-imbalance residual，因此把 isolated 与 concurrent 两部分分开识别。

令 $B_t\ge1$ 为 single-team internal dilation，$S_t\ge B_t$ 为 team width $t$ 在满 rank
fixed cohort 下的总 dilation，$C$ 为 rank cores，
$A(e)$ 为 event $e$ 中所有 active task 的完整 team-width 之和（截断到 $C$）。对 active task
$i$ 定义 peer occupancy：

$$
q_i(e)=
\begin{cases}
\min\left(1,\frac{\max(0,A(e)-t_i)}{C-t_i}\right), & t_i<C,\\
0, & t_i=C.
\end{cases}
$$

仅对 `cold_b/steady_b` GEMM phase 施加：

$$
D_i^{team}(e)=B_{t_i}+(S_{t_i}-B_{t_i})q_i(e),
\qquad
D_i(e)=D_i^{resource}(e)D_i^{team}(e).
$$

因此单 task 或无 peer 时达到 $B_t$，满 cohort 时达到 $S_t$，且不允许
$S_t<B_t<1$ 破坏 resource lower bound。$B_t/S_t$ 都是离散宽度表；未校准宽度返回 1，
不做跨宽度外推。该项在有真实 CPU placement 的 heavy event path 上按 phase 作用于 GEMM。
unplaced `dag_makespan()` 仍不加该项。production quick 的 $T_{iso}$ 与 LPT packing 不变，
但齐次形状之间的 makespan 比较乘上 occupancy-aware 整任务代理
$D^{team}(t,A)=B_t+(S_t-B_t)q(t,A)$，其中 $A=\min(\sum t_{\mathrm{lane}},C)$。
这只修正「孤立 $T_{iso}$ 选过宽齐次队」；不是 event 精度，也不恢复局部邻居序。
无表或未校准宽度时系数为 1。cold-phase CP-SAT 无法在线性 cumulative master 中精确表达随 active set 变化的
$q_i(e)$，而把 $B_t$ 或 $S_t$ 静态乘到整个 mode 会错误改变 wave 数与宽度的权衡。因此
analytical CP master 保持原 isolated cold surrogate；proof gap明确不包含 wide-team pressure，
不能声称证明完整并发 event objective。
完整 event rerank仍使用上式的 occupancy-aware $D_i^{team}(e)$，所以尾部 active teams 减少时
不会继续支付满 cohort 系数。

Arm-codex 80C 的校准使用三份真实 trace 中与最终 gate 不重叠的六层
（request008 L0/L10、request016 L10/L20、request022 L0/L10），固定 80 active cores、
4/8/16T、三种 deterministic order、early merge off、无 tail pool，每点 2 warmup/9 runs/
2 rotating weight copies。以各层、各 width 的三种 order 的
`median(measured/event-predicted)` 为样本，再跨六层取中位数，得到原始
4/8/16T=`0.920/1.303/1.470`；应用非加速下界后校准为
32T 因不能整除 80C，另用两个非 gate 层的 64-active-core `2x32T` 控制；40T 使用
`2x40T` full cohort。随后在相同两个 held-out 层分别只启用一个 8/16/32/40/80T team，得到
$B_8/B_{16}/B_{32}/B_{40}/B_{80}=1.144/1.251/1.494/1.424/2.306$，并取 $B_4=1$。
满 cohort 得到 $S_4/S_8/S_{16}/S_{40}=1.000/1.303/1.470/1.821$；32T 的
$q_{32}=2/3$，由残差中位 `1.577` 解得 $S_{32}=1.619$；80T 不存在 peer，取
$S_{80}=B_{80}=2.306$。这只能识别该机器、BF16 TP4 proxy、离散
4/8/16/32/40/80T cohort
域内的 residual pressure；route class、partial occupancy 和第二台 Arm 仍是 holdout 风险。

**jemalloc 重标定（v9，2026-09-19）。** 上述 $B_t/S_t$、1T/2T `by_width` 与窄 team
修正均通过无 workspace 的 Plan V2 调用测得，glibc 下含每次调用新分配 `route_out` 的
缺页与清零。改用 jemalloc never-purge 后以相同脚本、相同层与推导重测（服务速率与
topology v2 不变）：$S_{4/8/16/32/40/80}=1.000/1.192/1.276/1.516/1.512/2.029$，
$B_{8/16/32/40/80}=1.120/1.169/1.452/1.435/2.029$；1T `by_width` 为
$210.0\,\mu s+3.37\,\mu s\cdot R$，2T 为 $81.9+3.92R$；$c_1/c_2=0.717/0.507$。
同源 glibc 对照复现旧值（除 $S_{32}$、$B_{40}$ 漂移约 11%）。三个闸门层留出上，
event 模型实测/预测中位由 v8 的 0.898 变为 v9 的 0.984（中位 $|\log|$ 0.107→0.038）。
1T 的固定/每路由拆分不可良好识别（多次拟合固定项 100--210 µs），非拟合 peer 模式高估
约 5%。服务探针单次抽查显示 80T DRAM 可能低于 v2（未重测）。见
`tmp/jemalloc_recal_20260919/decision.md`；文件在
`bench_assets/moe_paper/arm_codex_numa3_80c_jemalloc/`，现有消费方未切换。

**探针校准的融合 event 模型（v10，Lab 候选，2026-09-19）。** v9 的三处已被实测否定：并发窄
lane 的 DRAM 需求（spill=1.0）比 DDRC 计数高约 6 倍；DRAM 服务"到 380 GB/s 才稀释"的饱和位置
不对（真实满载只有 44--71 GB/s 却已有减速）；$B_t/S_t$ 对单个大 M task 在 32T 上高估 24--28%。
v10 保留 phase 划分、事件模拟与 placement，删除 spill 规则、DRAM 饱和稀释、$B_t/S_t$ 与
$c_1/c_2$，并规定：**任何一项都由隔离该项的探针直接测量，不在 plan 时间上回归；整计划只用于验证。**
（先前一版在单目标实验格上回归响应形式，拟合格 MAE 0.026，但整计划开发集高估约 20%：单目标
bench 中目标的权重装载段总与全部后台 lane 的同步起跑相撞。该回归结果只保留为消融基线。）

孤立时间：

$$
T^{iso}_{v10}(M,t)=(1+\varepsilon)\sum_p \tau_p(M,t)+O(t),\qquad
\varepsilon=0.065,\quad O(t)=10\,\mu s+2.5\,\mu s\cdot t\ (t\ge4),
$$

$\tau_p$ 为 v9 的 phase 基准时长；$\varepsilon$ 由单 expert 孤立格（4/8/16/32T，M 256--2048）
测得，$O(t)$ 由"单个 team 顺序执行整层"的对照测得（8/16/32T 两层，误差 $\le1.8\%$）。
1T/2T 保留 v9 的 `by_width` operator phase。

争用：task 的每个 phase 属于 **装载**（$L$：v9 的 cold-B，即 W13 与 W2 的首个 12-row panel，
首次读入该 expert 的权重）或 **稳态**（$S$）。事件 $e$ 中，除 task $i$ 自身外正在装载的核数为
$n_L$，处于稳态的核数为 $n_S$。四条服务曲线由探针直接测得（$t\in\{2,4,8,16,32\}$，
$n\in\{0,8,16,32,48,64,76\}$，4T 后台 lane）：

| 曲线 | 目标 | 后台 | 测量量 |
| --- | --- | --- | --- |
| $D_{LL}(t,n)$ | M=12 expert 链（整段即装载） | M=12 链 | 目标链每 expert 跨度 / $n=0$ |
| $D_{LS}(t,n)$ | 同上 | M=2048 链 | 同上 |
| $D_{SS}(t,n)$ | 两个 route 数的单 expert | M=2048 链 | 每 route 斜率 / $n=0$ 斜率 |
| $D_{SL}(t,n)$ | 同上 | M=12 链 | 同上 |

斜率法 $(T(M_b)-T(M_a))/(M_b-M_a)$ 消去同步起跑碰撞与装载段。组合规则在测量前冻结，无自由
参数：

$$
D_i(e)=1+\bigl(D_{xL}(t_i,n_L)-1\bigr)+\bigl(D_{xS}(t_i,n_S)-1\bigr),\qquad x\in\{L,S\}
\text{ 为 } i \text{ 当前 phase 的类别},
$$

曲线对 $n$ 分段线性、超出末点取常数，对宽度按 $\log_2 t$ 线性插值。Arm-codex NUMA3 80C 实测
（两会话均值）：$D_{SS}\le1.03$；$D_{LL}$ 在 $n=48/64/76$ 为 4T 的 1.32/1.64/1.90、16T 的
1.53/1.85（$n=64$）；$D_{SL}$ 为 4T 的 1.05/1.09/1.13、2T 的 1.13/1.27/1.38；$D_{LS}$ 对 2T/4T
$\le1.016$，对 8/16/32T 为 1.11--1.35（$n=48$）。即争用几乎全部发生在权重装载段之间，
稳态对稳态可忽略；20 个忙核以内四条曲线均 $\le1.05$。16T/32T 的装载曲线两会话相差 3--15%
（0.10--0.15 ms 的任务），记为不稳定点并仍用均值；三个背景覆盖率 <1 的点按冻结规则剔除。
机理检验未通过：装载段每 expert 的核时超额 $(D_{LL}-1)\tau_L t$ 在 $n=48$ 为
0.39/0.60/0.89/1.23/2.09 core-ms（2/4/8/16/32T），不重合，因此曲线按"宽度分档的服务表"
报告，而不是单一硬件带宽曲线。同核数下后台 lane 越宽施加的负载略低（16T 后台比 4T 后台的
超额在 $n=64$ 小约 20--40%），模型只按核数计，不区分后台宽度。

整调用时间 $T_{call}=\text{makespan}+t_{over}$，$t_{over}=0.78$ ms 由 W 组协议（ready-token
merge 开、producer-hot 输入）下成对的"不追踪 wall / 追踪 compute makespan"直接测得（追踪只
推迟调用结束，不改变计算跨度：追踪调用的 native e2e 与不追踪 wall 相差 <0.2 ms）。

Stage window：选择取 V3 表（M2 的 N 组在 32 个未见 M 格上 regret $\le3.65\%$）。时间尺度随
负载变化（M2 的 C 组：同一 window 的收益从孤立 2% 到满载 21%）：windowed task 孤立时为
$(1-g_0)\tau$，$g_0=0.02$；在表的校准负载（同宽同 M 满载，模型超额 $e_{cal}$）下为
$r(t,M)\,\tau(1+e_{cal})$；其间按当前超额 $e/e_{cal}$ 线性过渡，超过取 $r$。无拟合参数。
静态选择在小 M/大 M 后台下最多损失 2.7%（C1 未过），本轮不处理。事件模型因此与 quick 使用
同一套 window 时间。

不进入预测的量：按宽度的残差 $\eta_t$ 只作诊断报告；DRAM 字节不作为时间模型的输入（孤立
4T team 以 21--27 GB/s 持续读 DRAM 而不受时间惩罚，满载每 lane 仅约 3.4 GB/s，该现象未解释）。

拟合外检查（未参与任何标定）：layer 12/29 的 54 个 full-stripe 开发整计划上 regret@1 为
0/0/0.08/0/2.70/0%，Spearman 0.85--1.00，wall/预测 中位 1.076（齐次）与 1.044（热点固定），
53/54 落在 $[0.85,1.15]$（v9：layer 29 regret 3.6--8.6%，热点固定计划 0.79--0.94）。先前采集的
68 个单目标负载格减速 MAE 0.054（无争用基准 0.115）：大 M 目标 0.02--0.035，小 M 目标在
同步/小 M 后台下低估 0.13--0.43（最大为 2T M144）。验收以 M2 的 W 组与 S 验证格为准
（`tmp/m2_validation_20260919/decision.md`），预测文件在测量前冻结。实现为 Lab 模块
`tmp/v10_model_20260919/v10.py`，不改 production、Plan V2、kernel 与默认消费方；探针记录见
`tmp/v10_probes_20260919/decision.md`。

验收结果（M4，2026-09-20，预测文件冻结后一次性测量，18 个未见层工作负载 × 11 个 full-stripe
候选，两会话）：W1 通过，实测/预测中位为齐次 1.072、热点固定 1.05、混合 1.075（模型整体偏低
5--8%，未修正）；W2 按冻结规则未过，regret@1 中位 0%、17/18 $\le5\%$、最大 5.003%
（差 0.003 个百分点），regret@2 最大 3.4%，失误集中为"8T 型层上偏好 p16_r4"（损失 4--5%）；
对照 v9 event 最大 10.35%、冻结 quick cost 最大 15.09%、无争用 10.35%。S 验证格：S2 通过
（减速 MAE 0.030，对无争用 0.062、v9 0.043），S1 未过——大 M 目标（M 768--1536）MAE
0.007--0.016，小 M 目标（4T M48）负载下低估至 0.40。按冻结规则采用收窄声明：模型把候选剪到
2 个（其最优与实测最优相差 $\le3.4\%$）并由实测共识 tuner 决定；争用层按"实测服务表、对大 M
任务成立"报告。不开第二轮修订；v10 仍为 Lab 候选，消费方未切换。

**v10 接入与模型目标 LNS（2026-09-20，opt-in，默认不变）。** 用户决定（2026-09-20）：将 v10
接入正式代码观察效果，按结果修正模型或搜索。实现：`cost_model/probe_event_model.py` 的
`ProbeEventModel` 以上述公式实现 `dag_makespan_placed`/`explain_dag_placed`/`T_iso`/
`call_time_placed`，phase 骨架取包装的 `AnalyticMoeCostModel`（v9 标定），其余 planner 接口
委托给它；native planner 导出被关闭（native 路径使用自身的 v8/v9 服务模型，会绕过该目标）。
标定文件 `bench_assets/moe_paper/arm_codex_numa3_80c_jemalloc/probe_event_v10_20260919.json`
即冻结的 `params_v10_probe.json`。给定 window policy 时，每个 task 的 window 由 lowering 同一
张表决定，事件模型因此看到与运行时相同的 window。事件模拟中曲线按整数核数预制表，并以装载/
稳态核数计数器增量维护，数值与 Lab 实现一致（306 个 M4 计划相对偏差 $\le 6\times10^{-15}$），
单次 220 task 评估 8.6 ms。quick：`IntervalPlanner` 若模型提供 `quick_homogeneous_scale` 则用它
代替 $B_t/S_t$ 占用比例；v10 返回 1，即 quick 按 v10 孤立时间 × window 时间尺度做齐次 LPT。

模型目标 LNS（`planners/model_lns.py`，离线参照物）：计划为一组串行 lane，lane 宽度
$t\in\{2,4,8,16,32\}$、位于单个 LLC 域内（首次适配递减装箱，v10 只读宽度与并发，不读物理
位置，因此任何合法装箱等价），目标为 v10 event makespan。移动：迟 lane 的 expert 迁到/换到早
lane（relocate/swap）、宽度拆分/合并（split/merge，孤立 LPT 重分）、复合重装（repack：迟 lane
与随机 1--7 条 lane 的 expert 合池，在同核数的宽度模板——齐次或一宽多窄——上孤立 LPT 重分；
P2 中单步移动够不到的混合形状由它覆盖）、lane 内顺序（降序/升序/热点先行其余升序）。每轮
评估一批采样邻居，改进则接受；连续 patience 批无改进时从最优解扰动重启。它不是 native
planner 的替代，而是 quick 的对照标尺与特征来源。效果实验 E1 的设计与结果见
`tmp/v10_integration_20260920/decision.md`。

**v11：中 M 服务探针修正（2026-09-20，Lab 候选，opt-in）。** E1（`tmp/v10_integration_20260920`）
显示 v10 驱动的搜索利用模型误差：含 2T lane 的搜索计划实测/预测 1.325、无 2T 的 1.175，锚点
1.09--1.11；逐 task trace（E1d）定位到负载下窄 lane 的中小 M 任务（2T M25--384 为 1.19--1.41，
4T M13--192 为 1.12--1.26），孤立时间不是原因。v11 在 v10 上加三项，均由探针 P6 直接测得
（`tmp/v11_probes_20260920/decision.md`），组合规则测量前冻结，不在整计划时间上拟合：

1. 孤立校正 $c(t,M)=T_{meas}/T_{v10}$（A 组 $n=0$ 每 expert 跨度，$M\in[24,384]$），孤立时间与
   per-expert 开销乘以 $c$；$M$ 上 log-log 插值，超出实测范围 1.5 倍后取 1，32T 不校正。
2. 按 $M$ 索引的稳态曲线：相邻实测 $M$ 对的每 route 斜率在装载/稳态背景下相对孤立斜率的比值
   给出 $D_{SL}(t,n;M_{mid})$、$D_{SS}(t,n;M_{mid})$，$M_{mid}=\sqrt{M_aM_b}$；v10 曲线作为其自身
   $M_{mid}$（2T 362、4T 724、8--32T 1448）处的点；稳态 phase 按任务 $M$ 在 $\log M$ 上插值。
3. 中 M 背景的装载等价：处于稳态 phase、路由数为 $M$ 的任务按 $w(M)\cdot t$ 计入装载核数，
   $(1-w(M))\cdot t$ 计入稳态核数。$w$ 由 B 组（4T M12 装载目标旁的 M 链背景）把实测 $D$ 反演为
   等价装载核数 $n_{eq}$（反演 $D_{LL}(4,\cdot)$），$w=\mathrm{median}_n\,(n_{eq}-f_L n)/((1-f_L)n)$，
   $f_L$ 为背景链的模型装载时间份额；实测 $w(24,48,96,192,384)=0.79,0.56,0.35,0.18,0.11$，
   $w(2048)=0$。装载 phase 仍按 $w=1$ 计。核数计数因此可为小数，曲线按 1/4 核制表。

window 的标定负载条件同样按 $w$ 拆分。装载曲线 $D_{LL},D_{LS}$、$\varepsilon$、$O(t)$、$t_{over}$、
$g_0$ 与 v10 相同；无 v11 扩展字段的标定文件与 v10 数值一致（306 个 M4 计划 $\le6\times10^{-15}$）。
偏离冻结文本一处（已记录）：11 个由小路由差斜率得到、低于 1 的稳态点取 1。描述性检查（E1 已见
数据，未参与构建）：无 2T 搜索计划实测/预测 1.088--1.097，与锚点一致；2T 计划仍为 1.15--1.16；
E1 七个计划中的选择 regret 中位 0、最大 7.9%（v10 为 7.7%/10.6%）。残差：2T M97--384 任务
约 1.25，8T M$\le$24 约 1.15--1.29；探针背景均为 4T lane，而背景 lane 越窄争用越重（v10 对照），
核数组合看不到这一点（候选原因，未检验）。验证见 E2（`tmp/v11_validation_20260920/decision.md`）。

模型目标 LNS 的补充（同日）：新增 rebalance（晚/早两条 lane 的 expert 合池后按孤立负载最小最大
做分支定界重分）、recreate（随机化贪心重建）、更多 lane 内顺序与阈值接受；搜索时限在最好的
至多 4 个不同起点间均分，各自下降（多起点）。在 P2 的 36 个可精确求解小实例（v10 目标）上，
单起点版本最大差距 2.0%，多起点版本全部达到精确最优（最大 0.004%）。

**v11 验收与快速 planner（E2/E3，2026-09-20）。** E2 在 18 个从未使用过的层上一次性测量：v11
搜索（lane 宽度限 $\ge4$）的计划在 17/18 个层上实测最快，比生产 quick 快 13.2%（中位，区间
$-14.9\%$ 到 $-5.5\%$）、比四个锚点构造中最好的快 10.2%、比 v10 搜索的同族计划快 3.7%；v11 在
7 个非 2T 计划上的 regret@1 中位 0%、最大 7.9%、regret@2 最大 1.8%（v10 为 3.4%/14.8%/5.5%）。
按 E2 冻结规则，2T lane 移出参照搜索空间：其计划实测/预测 1.237（锚点 1.085），预测与实测的
快慢符号一致 0/18。限制说明：P1--P6 的背景一律为 4T lane，而模型只按核数组合，背景 lane 越窄
争用越重（v10 对照已见），因此全 2T 计划被系统性低估约 14%；该现象未解释，留作 future work。

快速 planner（`planners/hot_wide_planner.py`，从参照计划中提取的三个特征）：形状取"少数宽 lane
（8/16/32T）承载最热 expert + 其余核全为 4T lane"的 14 个模板；lane 负载用孤立（含 window 尺度）
时间乘按宽度的事件/孤立比 $\{4:1.119,\,8:1.065,\,16:1.045,\,32:1.032\}$（由参照计划在 v11 下统计
得到）；lane 内顺序为"最大 expert 先行、其余按 M 升序"。在异构 lane 上做一次 LPT，取标定负载
最小的模板，请求路径上不跑事件模拟（Python 预热后约 30 ms，现有 Python quick 为 39 ms）。
模型目标下（E1 的 18 个工作负载、v11 评分）它与 75 s LNS 计划相差 $-0.3\%$（中位，最大 $+1.6\%$），
比生产 quick 低 8.8%。lane 内顺序这一项单独贡献约 2.5 个百分点（同一形状与装填下，降序
$+2.5\%$、升序 $+3.9\%$、热点先行升序 $-0.8\%$，相对 LNS）。硬件验收（E3，18 个新层）：较生产 quick 快 9.63%（中位，18/18），落后 90 s 参照搜索 3.82%（中位，最大 7.1%）；记录见 `tmp/fast_planner_20260920/decision.md`。

**生产接入（2026-09-20，用户决定）。** 三处改动，默认行为随之改变，依据均为已实测：

1. window 表按机器注册。`StageWindowPolicy` 增加 `machine_ids`（为空表示按 shape 匹配，保持既有
   表的行为），`default_stage_window_policy` 增加 `machine_id` 参数，`IntervalPlanner` 传入
   `model.calibration.machine_id`。V3 表登记到
   `arm_codex_320c_numa3_80c_sve256_jemalloc_narrow_merge_v9`，因此该机器上的 lowering 现在默认
   发出 window。实测依据（M4，18 层两会话）：生产 quick 计划加 window 后快 1.61%（中位，区间
   0.53%--5.81%），同族其余计划 0.99%--2.92%。
2. `PlannedMoE` 增加 `search_mode="hot_wide"`：用 `HotWidePlanner` 出计划，形状缓存按模板重建
   （模板与装填对给定路由计数是确定的）。
3. `MoePlannerRuntime` 增加 `event_calibration` 与 `search_mode`，并提供
   `enable_moe_planner_fast(analytic, event, ...)`：cost model 为 `ProbeEventModel`（孤立时间来自
   v11），phase 骨架、shape 与字节仍来自解析标定。`ProbeEventModel` 因此实现 `T_iso` 磁盘缓存
   接口（identity 含 v11 标定文件摘要，与 v9 的缓存不互串）。

快速 planner 的实现成本：每层规划中位 1.85 ms（Python，预热，18 个真实层；模板按上次胜出者
优先并在超过当前最好值时提前放弃，结果与穷举模板一致）。native quick 路径仍为 v8/v9，未改；
在 native 实现之前，请求路径上的净收益要扣掉这部分规划开销。

**v12/v13：背景 lane 宽度与 2T 装载等价（2026-09-20，候选，验收见 E5）。** E2 把 2T lane 逐出
搜索空间时留下的未解释项，由 P7/P8 两组探针定位（`tmp/v12_probes_20260920/decision.md`）。

P7（背景 lane 宽度）：同样核数下背景 lane 越窄争用越重。4T 目标在约 76 个背景核下的装载减速为
2.01（2T 背景）、1.91（4T）、1.70（8T，72 核）、1.40（16T，64 核）。以 4T 背景为基准的超额因子
$f_{L},f_{S}$ 由此测得（2T 背景 1.10--1.65，8--16T 背景 0.53--0.90），v12 把它乘在装载邻居带来的
超额上，背景宽度取"其他处于装载状态任务的核加权几何平均宽度"。单独加 v12 无效：E2 的 2T 计划
实测/预测 1.237→1.234，决策质量不变。

P8（真实背景下的组合检验）：4T 及以上背景下 v11 的组合是对的，包括真实层路由混合背景
（实测/预测 1.03--1.10）；2T 背景下低估 18--49%，且随背景路由数增大而恶化（M12 1.14--1.18、
M48 1.35--1.40、M96 1.38--1.56、真实混合 1.43--1.49）。按冻结规则反演装载等价会在曲线末端
饱和（超过约 76 等价装载核后曲线为常数），说明需要两项同时成立：(a) 2T lane 的稳态段对邻居
等同于装载段（4T 上测得的 $w(M)=0.35$--$0.79$ 在 2T 不成立）；(b) 2T 背景下曲线水平本身更高。
两项一起施加可复现 P8 全部 2T 格：实测/预测中位 0.978（v11 1.374、只加因子 1.310），逐格
0.93--1.03。

v13 = v11 + P7 因子 + `loading_weight_by_width = {2: 1.0}`。在已测数据上（描述性，未参与构建）：
E2 的 2T 计划 1.237→1.144，含 2T 候选时 regret@1 中位 6.07%→0.00%（最大 11.58% 不变）；E3 决策
不变，两个锚点因宽背景因子上移 0.01--0.02。模型整体约 1.09 的水平低估仍未解释，与此无关。
E5 验收（18 个全新层，一次性测量）：W1/W2/W3 全部未过——`l13` 实测/预测 1.281（锚点 1.098）、
九个计划上的 regret@1 中位 11.95%（v11 为 4.45%）、与 `l13_no2` 的快慢符号一致 0/18 且实测慢
11.82%。按冻结采用规则保留 v11：**v11 仍是参照模型，2T lane 仍在参照搜索空间之外**，v12/v13
作为"已测得但不充分的修正"记录。v13 能复现 P8 的探针格（0.978）却不能使整计划可预测：探针格
只有一种齐次背景，而计划里 2T lane 与宽 lane 混合、路由数各异，按 lane 宽度的单一装载权重不是
完整机理。生产路径不受影响（快速 planner 只读孤立时间，v11 与 v13 相同）。
记录：`tmp/v13_validation_20260920/decision.md`。

**标定域与可信宽度（2026-09-20，E8）。** 探针 P1--P8 只覆盖 lane 宽度 2--32T，`ProbeCurves`
对范围外的宽度夹到边界，因此 1T/40T/80T 的打分是外推。实测后果：`plan()` 的 tail-pool 候选
自动选中 1T 池（93/70/14 个池化任务），其池化任务实测跨度是预测孤立时间的 3.25--4.25 倍
（p90 5.69），尾部因此拖长 2.4--6.1 ms，使整计划比搜索结果慢 1.0--1.3 ms，而 planner 的
tail-pool 打分却认为它快 0.7--3.3%。修正不在 tail-pool 模拟，而在标定域：模型给出
`calibrated_widths`（曲线实测集）与 `reliable_widths`（再扣除整计划反证的宽度，v11+ 资产将
2T 列入 `unreliable_widths`，依据 E2/E5），`IntervalPlanner._tail_pool_candidates` 只在可信宽度
上自动生成候选（显式指定仍可用），`ModelLnsSearch` 默认取可信宽度。施加后 r008_l21 的最优
tail-pool 候选由 25.79 ms（2T 池）变为 26.74 ms（4T 池），高于最优严格候选 25.99 ms，planner
不再选它——与实测一致。

**路由切片（2026-09-20，E8）。** `ModelLnsSearch` 增加 `slice`/`unslice` 移动（等长 route
range，与运行时的 lowering 语义一致），默认关闭。在本工作的 54 个已测层上它没有价值，原因是
结构性的：最热 expert 在最宽 lane 上的时间与工作量下界（总孤立核时/80）之比中位 0.38、最大
0.44，没有任何一层由单个 expert 定界，切分它不能降低 makespan；打开该移动后搜索一次也不接受。
只有当单 expert 时间超过工作量下界时才需要开启，对这些 2048-token TopK6 请求约需 4500 条路由，
实测最大为 2034。仓库原有的 bounded tail repartition 家族在此机器上不可用（宽度表只为 96 核
标定，且要求计划恰有两个终端任务）。

**Diverse shortlist 与 lowering。** proof 与 pool 使用两个独立 solve。proof 在长时限内只
输出完整 master 的 $U_{CP},L_{CP},gap_{CP}$；pool 使用较短 root，不让更长 proof 偶然改变
候选邻域。pool root 后固定非热点 expert 的 width/domain 和 surrogate interval，保留 route
数最大的 16 个 expert 为 repair set；这些 expert 可重新选择 width/domain/start。每次得到解
$k$ 后禁止完全相同的 per-expert start vector：

$$
(s_1,\ldots,s_E)\ne(s_1^k,\ldots,s_E^k),
$$

并要求后续解的 surrogate makespan 不超过 pool root incumbent 的 $1+\epsilon_s$，重复至
32 个解或时限。因此相同 width/domain vector 的不同 order 是合法的 diverse plan；报告同时
保留 width/domain 与完整 schedule signature，不能把 pool 中 32 个解都解释为全局
$gap_{CP}$ 内的独立证明。每个解再固定 width/domain，以无 internal wait 的 service duration
建立较小的二维 no-overlap lowering。第一步固定 fluid master start，只求 contiguous core
placement；若不可行，第二步才允许 task 不早于 fluid start 地最小延迟，并以 lowering makespan
为目标，状态显式标为 `*_DELAYED`。相邻 core 使用者形成 runtime dependency；lowered start
time 不下沉为主动 idling/release gate。该 fallback stretch 必须计入 lowering 状态与耗时，
且 root proof gap 仍只属于 fluid surrogate。随后用完整 analytical event model（包含
current cache/refill、LLC/DRAM contention 和 deterministic window）重排，硬件只测前 8 名。

第一版必须分别报告 root CP gap、32 解生成数、lowering 成功数、event ranking、前 8 名实测
regret、相对 legacy full/one-step gate/fixed controls 的回退、resource-LB gap，以及
$T_{plan}+T_{execute}$。验收条件为 root gap 不超过 1%（资源有限时可明确放宽到 5%）、
measured shortlist regret 不超过 5%、三个真实 trace 相对 one-step gate 均不回退超过 2%。
通过后才扩展到完整 43 层和第二台 Arm；该 solver 始终为离线 paper oracle，不改变
production quick/full 默认、Plan V2 schema 或 native ABI。

### 6.6 Production cold search 的 native 并行编码

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
profile 的 $\widehat I_i,\widehat D_i$ 仍只描述 expert compute；production planner
固定 `early_merge=true` 是独立于模型目标的执行 policy，不代表模型已经估计 combine
收益。在增加 merge service time、expert/merge 异构 contention 和真实分布留出验证
前，不得把该固定 policy 解释为跨 workload 或跨机器的最优性结论。

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

quick 与 full 的目标和预算明确分离。quick 只比较 homogeneous team shape，以
isolated $T_{iso}$ 和 LPT lane load 给出有界延迟的较优方案；不运行 phase-DAG、
mixed-width、temporal-order 或 dynamic-tail 搜索。full 枚举模型支持的全部 strict
shape（包括 mixed-width 和 temporal order），再从有竞争力的 strict head 派生
whole-expert tail pool 与合法 bounded tail repartition，并用 phase-DAG 评分。full
因此寻找完整模型可行域内的最优或不可由校准误差区分的近优方案，而不是 quick 的
别名；代价是显著更高的 cold planning latency，定位为离线分析或显式冷规划。

对未显式传入 `shapes=` 的 analytical full search，homogeneous quick 胜者 $q^*$
也作为一个 LPT strict candidate 加入完整候选集合。解析校准的 relative error 是
共享公式、服务曲线和 residual 的系统误差，不是每 wave 独立采样噪声，因此其绝对
不确定性为 $U_c=\epsilon\hat T_c$，不按 `profile_runs` 或 wave 数的平方根缩小。
empirical full-call anchor 的采样误差仍沿用 profile-specific 估计。

设重评分后的 quick 基线为 $(\hat T_q,U_q)$，候选 $c$ 为
$(\hat T_c,U_c)$。先取完整模型空间中的期望 makespan 最小者：

$$
c_0=\arg\min_{c\in\mathcal C_{full}}\hat T_c.
$$

令 $w(c)$ 为 candidate shape 的最大 team width，并令不确定区间
$I_c=[\hat T_c-U_c,\hat T_c+U_c]$。当 $w(c_0)>8$ 时，只考虑与 $I_{c_0}$ 相交且
最大宽度不超过下一个已校准窄档
$w^- = \max\{t\in\mathcal T:t<w(c_0)\}$ 的候选集合 $\mathcal O^-$。若
$\mathcal O^-$ 非空，则

$$
c^*=\arg\min_{c\in\mathcal O^-}\hat T_c;
$$

否则 $c^*=c_0$。该 gate 最多降低一个宽度档，不会从 32T 连续退到 8T；当 winner
最大宽度不超过 8T 时也不启用。它针对当前模型尚未携带 interval-to-LLC placement、
因而在系统误差内低估 wide mixed-team 并发压力的边界。它不恢复旧的最小 active
working-set tie policy：该策略曾把模型第一名换成实测明显更慢的
`(32,32,8,8)`。在同一最大宽度档内仍以 expected makespan 为主，数值相同才依次使用
pessimistic time、active working set 和 resource group 作确定性 tie-break。
显式 `shapes=`、empirical profile、stage-specific planner 和 forced pool 保持原选择
语义。Plan V2、cache key、runtime claim 和数值语义均不变。

Production quick 进一步把生命周期分成初始化和逐请求规划。初始化对部署声明的
$M_{max}$ 与合法宽度集合 $\mathcal T$ 生成并持久化

$$
\mathcal I=\{I(M,t)\mid 1\le M\le M_{max},\ t\in\mathcal T\}.
$$

逐请求 planner 只读取 $\mathcal I$、比较有限 homogeneous shapes、执行 LPT 并
materialize Plan V2，不再调用解析 `predict_expert()`。Route histogram 被视为请求
特有输入，production runtime 默认不缓存完整 route plan，也不构造 bucket signature；
磁盘 cache 只保存机器/模型 identity 绑定的 `T_iso` 标量。这样
`planner_initialization_ms` 与 `runtime_plan_ms` 具有互斥口径。

部署诊断可设置 `FUSED_CPP_MOE_PLANNER_FIXED_THREADS=t_f`，将 routed-expert quick
候选域退化为唯一 homogeneous shape
$\left(t_f,\ldots,t_f\right)$，随后仍按 route 降序执行相同 LPT。`0`/未设置保留
多宽度 C++ quick；该控制不进入 full 或 synthetic shared-expert 候选域。

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

#### 8.2.4a Workspace isolated 阶段候选与离线消融（Lab，2026-09-07）

冻结 v8 不变。`reaccount_workspace_phase_model.py` 仅生成独立候选：
\(G(M,t)=a_t+b_t\lceil M/t\rceil\)，\(a_t\ge0,b_t>0\)；
\(T_s(M,t)=\max(f_{s,t},k_{s,t}T^{physical}_s(M,t))\)，
\(f_{s,t}\ge0,k_{s,t}>0\)，\(s\in\{W13,W2\}\)。复用已有
gather/stage calibration 语义，不改变资源公式、schema、native 或 production。
候选将 expert fixed/route/by-width operator residual 全部归零，因而旧 total
residual 不再通过 gather 分支重新分摊到 GEMM。只拟合 workspace isolated
session1 的阶段时间，不读取 full-workload 时间或总时间作为 fit target；
session2 是同 shape 重复性检查。M1 完全不拟合，是已知反例的外推检查而非全新
prospective holdout。8T 只有一个 M 点，floor/scale 不可充分辨识；不得宣称
全 route 区间准确。宽度仅支持实测的1/8/16，含2/4T 的完整计划明确跳过。

消融在 v8 和阶段候选上分别重跑完整 placed DAG：去掉 wide、narrow、两者，
去掉 wide 的 isolated floor 或 peer increment，再分别去掉共享 GEMM/L2/LLC/
DRAM/epilogue 放大或将 spill 固定到各 phase 的 isolated 值。共享资源消融保持
isolated service cost 不变，只在独立 Lab 单线程进程中屏蔽对应 event scale。
事件时序会改变，各项消融差值不可相加，亦不能直接解释为真实硬件因果贡献。

剪枝表与候选空间不变：本候选的 `Prune=Accept=empty`，旧 partial-order
calibration 不可绑定它。必须同时报告大 M/16T 与小 M/1T 的阶段增量及错误方向；
改善一类不能掩盖另一类退化。未完成独立 shape/width holdout 前，不替换 v8。

#### 8.2.4b A/B 供给账单与 phase-lifetime 重叠候选（Lab，2026-09-07）

`audit_ab_supply.py` 复算冻结 v8 的首 panel/steady A、B、C 账，不修改
`analytic_model.py`、calibration、production 默认值或 native parity。
每个 phase 验证 \(Q_A+Q_B+Q_C=Q_{L2}=Q_{LLC}\)；这个等式反映现有
endpoint-refill 口径，不表示真实 L2-to-L1 与 LLC-to-L2 物理流量相等。
L2 驻留的 B 仍有向 L1/register 供给，但当前 L2 资源没有独立记录该流量；
不能在未厘清 endpoint probe 语义前直接叠加全部 B load，否则也可能重复收费。

A 有效 load 字节与 touched cache-line footprint 分开审计。对对齐 panel，
K4 块 \(j\) 访问区间为
\([8jm_p,8jm_p+8m_c)\)，对所有块求 cache-line 集合的并集。
例如64-byte line、K4096、M1-exact 的 \(m_c=2,m_p=8\)，有效 payload16 KiB，
但 touched lines64 KiB。该值不等于实测 LLC miss，也未直接替换 v8 流量。

新增仅 Lab 的 `LifetimeOverlapModel`，沿用原 demand、placed spill、service
capacity 和可选旧 wide/narrow 项。对当前 phase 的无额外资源争用耗时 \(T_i^0\)
（保留 isolated 下界及选定旧项），按完整 phase 平均请求率计算每个 rank/domain
资源约束 \(r\)：

\[
U_r=\frac{\sum_i q_{ir}/T_i^0}{C_r},\qquad
f_i=\max\left(1,\max_{r:q_{ir}>0}U_r\right),\qquad
T_i'=f_iT_i^0.
\]

对每个约束，其所有 requester 都至少放慢 \(\max(1,U_r)\)，所以
\(\sum_i q_{ir}/T_i'\le C_r\)。缩放的是 phase progress，不是仅缩放一个
可能被 ECM max 隐藏的 transfer 项；因此避免简单替换 offered-rate 分母后
破坏总容量约束。这个 fluid 近似不表达 burst arrival 或 DDR queue latency，
并非声明旧 service-interval 定义有实现 bug。zero-demand setup 不因该约束放慢。

验证采用25个账单 shape、7个已知 workspace 固定计划的28次 placed replay，
比较保留/清零旧 wide+narrow 与原/lifetime 分配的交叉消融。不拟合数据。
须报告 small-M 阶段反例与 median 顺序，不用总误差改善掩盖退化；两次硬件
session 的中位方向一致也不等于通过 actionable-margin/置信区间 gate。
候选 `Prune=Accept=empty`，不绑定旧 partial-order report、不接搜索。
结果与局限见 `optimizations/fused_moe_sve/results/ab_supply_ledger_overlap_20260907.md`。

#### 8.2.4c Cache-line 与 private-endpoint 分账候选（Lab，2026-09-07）

`burst_endpoint_model.py` 仅支持固定 H4096/F512/SVE256 full-stripe 诊断。
实际 A/B 地址归属 refill 未观测，字段保留 null；已有 PMU 仅按 victim 汇总，
不从总 counter 相减伪造逐 tensor 流量。A payload、touched lines、模型 refill
估计分开保存。每个 panel 的64-byte line 集合沿用8.2.4b计算，记为 \(L_p\)。
候选以 \(Q_{3,A}=t\sum_p L_p\) 替代有效 payload 的 owner 聚合 refill 估计；
B 的下层 refill 继续沿用冻结的 L2 驻留规则，未新增命中率拟合。

私有 L2 endpoint 还须供给缓存命中的数据。候选用显式几何近似：
当 \(L_p+B_{tile}>C_{L1}\) 时，A 每个 N tile 重扫，否则每个 owner 一扫；
owner B 加 A panel 超过 L1 时 B 每个 M panel 重扫，否则仅首 panel 收费。
这只是 L1 驻留假设，不是完整替换模拟器；不含未知 write-allocate/RFO。

\[
Q_2=\max(Q_3,Q_{2,A}+Q_{2,B}+Q_C),\qquad
T_2=Q_2/R_2(t),\qquad T_3=Q_3/R_3(t).
\]

L2 是 private-team endpoint 下界，不再作为 rank 间的第二份共享请求；
通过独立字段保存 \(Q_2\)，共享 resource demand 中 L2为0，但 \(T_2\) 保留。
GEMM body 仍为 \(\max(T_{core},T_2,T_3,T_{DRAM})\)，endpoint 时间不相加。
LLC/DRAM、cold/gather、domain injection 保留原 service-interval burst 计算，
绝不使用8.2.4b已拒绝的 phase-lifetime 平均。stage floor、scale、旧 wide/narrow
均原样保留，并以清零旧项的独立消融检查依赖。

源码核对表明 service sampler 的 L2/LLC是 shared-read-only B-only 循环，
计费单位为 B scan payload，包含到寄存器的完整路径；cold DRAM靠互异 B池轮换，
没有新 workload 的逐轮 scrub。该定义不证明采样时每次访问都来自命名层，
也不证明旧曲线可直接推广为新 A-pattern 或 disjoint-owner 的精确服务率。

35-shape isolated账与42次已有 workspace placed 回放表明：新的私有 L2约束
仍被 compute隐藏，line-only与完整候选在这7个计划上的预测相同。保留旧项时
compute-end MAPE20.24%→20.39%，M1/1T阶段MAPE38.79%→38.71%，两场中位
方向一致的8对中正确数6→5；未通过精度/排序采用条件。该改动完成分账与
burst-preserving诊断参考，不替换v8，不改生产/native/schema或candidate空间，
`Prune=Accept=empty`。原有通用 stage-window explanation/native export不适用于
该Lab候选，显式拒绝。报告：`optimizations/fused_moe_sve/results/burst_private_endpoint_20260907.md`。

#### 8.2.4d 小 T isolated 阶段限定（Lab，2026-09-08）

`small_t_isolated.py` 将已有校验数据限定到1/2/4T，共29个shape/width、87个
gather/W13/W2阶段点，各两场中位数。只用第一场且M不为1/7的点拟合；M1保留
guard，M7是历史validation，不声称新的prospective holdout。完全不读并发计划
作为fit/evaluation，不拟合operator总时长，也不运行placed event。

GEMM复现已存在的非负affine形式 \(T_s=f_{s,t}+k_{s,t}P_s\)。冻结物理基底
按每个cold/steady/setup phase分解为
\(P_s=\sum_p(C_p+\max(X_p-C_p,0)+E_p+O_p)\)，其中
\(X_p=\max(T_{L2,p},T_{LLC,p},T_{DRAM,p})\)。额外启动项 \(f\) 单列，
不生成memory demand；\(k\) 是effective stage scale，不能解释为独立拟合的
compute/A/B bandwidth。8.2.4c的新endpoint账仅作旁列诊断，不能静默替换拟合基底。

Gather独立使用 \(f_{g,t}+b_{g,t}\lceil M/t\rceil\)，不拿operator residual
充当gather。对相同可见特征下的观测中位数最小/最大值 \(a,b>0\)，任意固定
预测至少有一项相对误差不小于 \((b-a)/(b+a)\)，最优常数为 \(2ab/(a+b)\)。
这只是匹配既有观测的下界，不是未来真实硬件误差的统计下界。M1/2T gather两场
12.97/29.22us使该下界为38.52%；M7/M8在2T的ceil特征相同，合并观测下界28.14%。

复现结果：GEMM历史M1 guard和M7 20%检查通过，但W13/M7平均误差5.76%→10.13%，
并非所有M改善。Gather的1/2T M7失败，故完整isolated模型不合格。stage-only
估计器明确拒绝`T_iso`/planner总时长导出，不扩宽度、不加并发项、不替换v8；
`Prune=Accept=empty`。报告：`optimizations/fused_moe_sve/results/small_t_isolated_stages_20260908.md`。

#### 8.2.4e Exact-M W13 条件供给响应（Lab，2026-09-08）

固定H4096/F512，M1/4/8/12×1/2/4T，B scrubbed/preloaded×0/4/8个reader，
两场各31轮。`bench_kernel_response.py`复用生产JIT但不修改生产；分别测真实
kernel与B-only供给控制。仅第一场peers0两种缓存状态拟合：
\(\beta_{raw}=(T_c-H)/(D_c-D_h)\)，
\(\widehat T(D)=H+\operatorname{clip}(\beta_{raw},0,1)\max(D-D_h,0)\)。
这里H是B-preloaded真实kernel时间，不是纯compute；D是独立实测B-only时间，
不是plan-visible pressure或DDR latency。B stripe为8/4/2MiB，均超过1.25MiB
私有L2；预热不等于L2-hit，scrub也不证明所有请求都来自DRAM。

冻结第一场参数验证两场peers4/8；第二场重新拟合仅检查稳定性。预设门槛为
供给分离≥10%、两场raw beta∈[0,1]、漂移≤0.15、holdout MAPE≤5%、最大≤10%、
不劣于常数H，clipping不豁免raw beta。12组仅3组通过。平均误差2.008%，
同缓存状态isolated lookup为2.252%；M1/1T仍有17.074%最大误差。额外诊断
\(T=F+C+D-\alpha\min(C,D)\)使用旧compute proxy，3组不可辨识、6组参数
越界，不能把响应系数解释成计算/访存百分比。

冷B isolated第一场查表预测第二场同shape，MAPE0.437%（v8阶段7.499%），
仅证明已测shape的重复性，不证明未见M插值、完整阶段wrapper或并发预测。
下一步限定exact-M/kernel-family基准的未见M验证；不扩W2、不换v8、
不替换wide/narrow、`Prune=Accept=empty`。报告：
`optimizations/fused_moe_sve/results/kernel_joint_response_20260908.md`。

后续压力曲线（同日，独立数据）：M1/M12×1/2/4T，cold B，0/1/2/4/8/12/16
reader，两场各31轮，同时记录DDRC与B-only供给。16 reader时domain27实测读流量
约72–76GB/s，1T B-only时间增加约25.6%；M1约增加33/13/10%，M12约增加
1–1.7%。M12最高档小幅损失可重复，但没有两场均>2%的operational crossing，
不能据此拟合物理拐点或证明已变为memory-bound。DDRC flux包含victim与后台，
occupancy/command只是计数器单位的queue特征，不是victim latency纳秒；低reader
档位非单调，不能将count当带宽。此后续仅诊断，不修改模型公式或剪枝。
报告：`optimizations/fused_moe_sve/results/memory_pressure_curve_20260908.md`。

压力响应拟合（Lab）：冻结第一场零reader的\(T_0,D_0\)，
\(p=\max(0,D/D_0-1)\)，比较
\(\widehat T=T_0(1+ap)\)与\(T_0[1+a\max(0,p-p_0)]\)，\(a\ge0\)。
D是独立实测B-only时间，a不是访存占比。第一场六个非零reader档逐档LOPO；
零档是共同校准anchor，不声称将零档独立留出。按cell相对时间平方误差拟合。
阈值只有LOPO MAPE改善≥10%且≥0.1百分点、max不恶化、各折正阈值、阈值span
≤全训练p范围上限的20%、斜率CV≤25%才保留。选型仅第一场，序列化后才加载第二场；
第二场连victim/supply anchor也不重新归一化。全部六组选择linear。第二场M1
平均/最大误差7.516/25.526%→0.896/1.871%，M12为0.538/2.084%→0.140/0.690%。
M12/1T阈值LOPO有0.088百分点的小改善且参数稳定，但不满足复杂度收益门槛，不能
误称所有阈值都不稳定。六个第二场最高档p略超第一场范围，显式标记且不排除。
两场此前已观察，属于历史回放，不是新盲测；不识别plan→pressure映射、不导出
生产profile、不改v8或剪枝。报告：
`optimizations/fused_moe_sve/results/pressure_response_fit_20260908.md`。

冻结响应的新数据验证（同日，参数冻结后采集）：M1/M12×1/2/4T，两场各31轮，
新的3/6/10/14 reader作为主验证，0/16档仅作独立控制。固定a、T0、D0，不重新
拟合、选阈值或设anchor；二进制与原校准一致。每shape/session要求新四档
MAPE≤2%、max≤5%、且优于isolated常数，11/12通过。M1汇总平均/最大误差
9.709/25.733%→0.749/2.347%，M12为0.424/1.303%→0.194/0.445%。第一场
M12/1T平均误差0.322%略劣于常数0.267%，不隐藏该失败。只验证已知shape与
同类synthetic background下的未见count，不证明未见M、混合计划或plan→pressure。
不改生产profile、v8或剪枝。报告：
`optimizations/fused_moe_sve/results/pressure_response_prospective_20260908.md`。

真实W13后台迁移（同日，Lab）：固定16个1T后台核心，分别运行M1、M120（10个
M12 panel）或8+8混合；无后台/reader为对照。两场各31轮，victim与后台输出独立
校验。扩展probe二进制但不改生产JIT；冻结原a/T0/D0，不选新阈值。对照12/12通过
≤5%最大误差门槛，真实后台仅5/12 shape/session通过原2%/5%且优于isolated要求。
M1汇总MAPE15.414%→2.027%（max7.998%），M12从0.247%恶化到0.664%。M1/1T
小/大后台B-only中位约334/335us、337/337us，真实victim约428/459us；逐轮配对
增幅7.994%/7.794%，B-only配对差异CI均跨0。不能直接将单一供给标量推广到
任意真实kernel竞争；同时显式保留1T全部真实压力超原拟合范围的限制，不声称已
归因到A、B、缓存或queue。旧reader模型保留，生产profile、v8与剪枝不变。
报告：`optimizations/fused_moe_sve/results/real_kernel_background_20260908.md`。

A-only/A+B对照（同日，Lab，M1/1T）：生产无对应load-only模式，因此仅在Lab
生成M1加载骨架，B-only版本与生产机器码逐字节一致；保留A地址/双缓冲加载顺序，
删除矩阵指令与输出。A逻辑payload8KiB、含配对padding的唯一请求16KiB、line覆盖
64KiB、64个Ntile重复请求合计1MiB，均不等于实测refill。两场各31轮，大/小后台
真实W13配对差+34.77/+29.29us；A-only为−0.33/−0.20us，AB load-only为
−15.21/−15.74us，方向相反；保留计算去掉epilogue/store仍为+38.90/+33.57us。
不支持新增正A供给惩罚来解释31us，亦不排除完整kernel中的A交互；不同probe
之差不是可相加纯compute/memory时间。只保留load/compute共同执行的待查方向，
不拟合、不改冻结模型、production/v8/剪枝。报告：
`optimizations/fused_moe_sve/results/ab_supply_contrast_20260908.md`。

计算/依赖对照（同日，Lab）：M1固定A/B加载，0/1/2/4/8条矩阵指令每K4，并比较
loaded4/resident4/serial4/NOP4。B与标准loaded4机器码均逐字节匹配生产；第一阶段
两场36-cell×31轮。切断矩阵指令对加载值的依赖后差距仍在，故补两场24-cell确认：
无A/B加载的同值寄存器矩阵约192.6–192.8us，小/大后台差−0.06/−0.10us；
AB+loaded4差38.53/37.62us、AB+resident4差38.96/33.52us，AB+整数/NOP仍为
负差。额外约10–12万cycles伴随同量级backend-stall增量；大后台LLC miss更多，
L2 refill近似不变，DDR总流量更低。证据局部定位到矩阵执行与访存流共存的
重叠/吞吐损失，不能指定某个load queue、端口、预取或DDR参数，亦不是可独立
加上的31us固定项。两阶段分别保留身份与波动，不拟合、不改v8/生产/剪枝。
报告：`optimizations/fused_moe_sve/results/m1_compute_issue_20260908.md`。

#### 8.2.4f 实测分层供给小模型（Lab，2026-09-08）

仅对双LLC M12后台数据的M1/1T W13作离线条件预测。定义实测事件特征
\(m=\mathrm{LL\_CACHE\_MISS\_RD}/\mathrm{L2D\_CACHE\_REFILL}\)，
\(Q=\sum\mathrm{DDRC\ occupancy}/\sum\mathrm{DDRC\ read\ commands}\)，
\(B=\sum\mathrm{DDRC\ read\ GB/s}\)。先逐cell逐轮求比值，再取31轮中位；
Q不是纳秒，m不是精确cache命中概率，B必须包含两组DDRC。
仅第一场无后台固定\(T_0,m_0,Q_0,B_0\)，不以第二场重新设anchor。

令\(l=\max(0,m-m_0)\)，\(q=\max(0,Q/Q_0-1)\)，候选为
\(\widehat T=T_0(1+\beta_q q+\beta_l l)\)，或增加
\(\beta_{ql}ql\)，所有系数非负。以cell相对时间平方误差拟合。
对照是isolated常数、单一带宽\(\max(0,B/B_0-1)\)、queue与local模型。
系数只代表此shape/协议下的条件响应，不分解为独立可相加物理服务时间。

第一场只使用无后台、same/other各8/24/39后台。按count同时留出两个placement
做训练内部LOCO；交互项仅当CV MAPE改善≥10%且max不恶化才保留。16/32后台
及全部balanced placement均不参与拟合或选型。序列化后再回放第二场所有点；
48/64/78独立报告为高压力外推，超训练特征范围显式标记。精度参考门槛为每组
MAPE≤5%、max≤10%，禁止用平均结果掩盖外推失败。

输入是目标执行期间的实测PMU事件，虽然排除cycles/backend stall/target time，
仍不是plan-visible前瞻模型或因果识别；历史数据此前已看过，非全新盲测。
只使用新的双LLC协议，不混入旧worker/scrub生命周期。生产v8与旧冻结响应模型
不变，`Prune=Accept=empty`。结果见
`optimizations/fused_moe_sve/results/layered_measured_supply_20260908.md`。

本轮训练内部LOCO选择交互式：MAPE由相加式7.42%降至1.57%，max由19.63%
降至5.30%。冻结系数\((\beta_q,\beta_l,\beta_{ql})=(0.101799,0.402751,0.208058)\)，
\(T_0=302.74\,\mu s\)。第二场留出count16/32的MAPE/max为1.54%/4.18%，
未训练balanced16/24/32为2.92%/4.48%；balanced48/64/78外推为11.70%/15.57%，
未过门槛。第二场整体3.22%平均误差不能豁免高压力失败。保留有界Lab参考，
不重拟合留出、不导出production profile、不修改v8或偏序剪枝。

#### 8.2.5 Full-stage packed-B 复用与 width-derived owner stripe

> Workspace floor identification 后续（Lab）：固定 M3/4/8/16/24 ×
> 1/2/4/8/16T，补 M62/1341 ×2/4T，M7 全宽度不拟合，M1 继续保护。
> 比较既有 max-floor 形式与仅 Lab 的非负 affine 阶段形式
> \(T_s=f_{s,t}+k_{s,t}T_s^{physical}\)。第一场拟合，第二场复现；
> M7 用于形式验证（若用于选择就不是最终独立测试），要求每个阶段两场
> 误差不超过20%，并且 M1 GEMM 不劣于 v8。报告参数跨 session 及
> leave-one-M-out 稳定性，不把低训练误差视为 floor 可辨识。未过门槛时
> 不替换 wide/narrow 项，production schema、默认值及剪枝不变。
> Lab affine adapter 把 fixed 部分作为零资源需求的独立 stage-setup phase，
> 只对原 GEMM body 应用 scale，不把 startup 再作为带宽需求或 wide-team
> 放大对象。隔离 probe 移走 lane 中间 task 时，原后继必须接回原前驱，
> 不能只等待已经被提前完成的目标；目标仍作为所有后台 task 的依赖。

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

#### 9.41 Synthetic shared expert 的有限 mixed-width quick plan

standalone/TP 且 shared 与 routed expert 使用相同 rank-local $(H,F)$ 时，把唯一 shared
expert 表示为内部 expert $e_s=E$。对每个 token 在原 TopK route 后追加
$(e_s,1)$，并将原 routed 权重乘 `routed_scaling_factor`；因此现有 FP32 route merge
一次完成 routed scaling、shared 相加和 BF16 store。该等价只改变新 combined API
的舍入位置，不改变 routed-only API；EP 中 shared 必须位于 routed all-reduce 之后，
不属于本适用域。

设 rank 有 $C$ 个核心，quick calibration 给出合法宽度集 $W$。首版候选为

$$
\mathcal S_{rs}=\{(s,r,\ldots,r):s,r\in W,\ r\le s,\ C-s\ge0,\ r\mid(C-s)\},
$$

其中第一条 lane 宽度为 $s$，其余 $(C-s)/r$ 条 lane 宽度为 $r$；$s=C$ 时只有
一条 lane，作为顺序 endpoint；该 endpoint 只在至多一个 active routed expert 时保留，
多个 routed task 时不搜索完全串行的全核计划。约束 $r\le s$ 排除给短 routed expert
比全 token shared expert 更宽 team 的候选。当
$\sum_{i\ne s}M_i\ge M_s$（正常 TopK 至少为 1）时还要求 $s\le\lfloor C/2\rfloor$：
routed 总 GEMM 工作不少于 shared，不能让 shared 独占超过一半的不可抢占核心。
shared task 固定为 lane 0 的首任务，初始 load 为
$T_{iso}(M=T,s)$。随后 routed experts 按 route 数降序，以

$$
\arg\min_l\left(L_l+T_{iso}(M_i,t_l),l\right)
$$

做确定性异构 LPT；shared 完成后 lane 0 可继续领取 routed task。候选 makespan
为 $\max_l L_l$，不运行完整 phase-DAG，也不引入实测 route 表。每个 task 的
W13/W2 window 仍由既有 $(M,t,H,F,n_{tile})$ 解析 policy 选择。

cache identity 在原 route histogram/signature 后增加 synthetic-shared mode 和
$e_s$；同一 histogram 的 routed-only 与 combined plan 不得互命中。Plan lowering
仍产生 whole-expert strict Plan V2，故 schema、native task ABI、依赖语义和 worker
pool 均不改变。首版只支持一个 shared expert、SVE BF16 fused-SiLU、完整本地 expert
域和 standalone/TP；不同 $F$、bias、EP 与 clamped-SwiGLU 必须拒绝或走旧路径。

#### 9.42 Quick planner 的版本化 $T_{iso}$ 磁盘缓存

production quick planner 的候选评分只消费标量 $T_{iso}(M,t)$。解析模型首次见到
$(M,t)$ 时仍按本章定义完整构造 W13/W2 mapping、cache/DRAM demand 和 phase，并将
结果写入

$$
\mathcal C_{iso}[I_{model},M,t]=T_{iso}(M,t),
$$

其中 $I_{model}$ 包含 analytical model schema/name 与公式源文件 SHA256、完整 machine calibration、
$(H,F,E)$、standalone/TP mode 与 degree、并发 rank、backend N tile、exact-M policy、
down-output element size 和 supported widths。只有完整 identity 相等的文件才可加载；
任一字段变化都映射到不同 cache 文件。cache 只存有限正数的标量时间，不序列化
`ExpertPrediction`、route signature 或最终 Plan V2。

默认目录为 `~/.fused_cpp/cache/moe_costs`。runtime 初始化时只读匹配文件，quick
planning 对未命中点执行原解析公式；首次调用结束后在文件锁内合并已有点，并以临时文件
加 `os.replace` 原子写回。文件缺失、损坏、identity 不符或不可写都降级为进程内解析，
不改变候选、排序、目标、Plan V2 或算子正确性。`cost_cache_dir=None` 可显式关闭。

#### 9.43 显式 gather 压力原型（默认关闭）

原 v8 把 isolated total residual 全部表示为 W13 前的零资源 `operator` phase。这能拟合
$T_{iso}$，却会把真实的 `gather_pack_a` DRAM 流量从并发事件中删除，并把非 gather
residual 错放到 W13 之前。scrubbed trace 直接显示 lane-head target 的 W13 与 peer
gather 重叠，因此增加一个 calibration 可选、默认关闭的原型：

$$
T_g(M,t)=f_g+r_g\left\lceil\frac{M}{t}\right\rceil,
\qquad
Q_g(M)=\alpha_g M H(b_{in}+b_{pack}).
$$

启用时，`gather_pack_a` 成为 active-thread 数为 $\min(M,t)$ 的显式 phase，平均请求率为
$Q_g/T_g$；它与 cold/steady GEMM phase 一同进入既有 event resource allocator。为不把
旧 isolated residual 丢掉，若旧 width-specific residual 为 $O(M,t)$，则
$\max(O-T_g,0)$ 按原 W13/W2 phase 时间比例分配回 compute phase。因而已有 residual-fit
width 的 isolated 总时间保持为旧 $O+T_{W13}+T_{W2}$，但时间位置和资源身份被修正；
未启用该字段的旧 calibration 逐位保持 v8 行为。

首个 Arm 原型取 trace-derived $f_g=5\ \mu s$、$r_g=2.9\ \mu s$/worker-row，扫描单一
effective-traffic 参数 $\alpha_g$。在 $M\in\{1,2,5,6,12\}$、五种 context 的同一批
25 个点上，$\alpha_g\approx3$ 将 absolute-relative-error mean 从 frozen-v8 的
21.7% 降到 12.6%，并使 M1 full/16T-only 达到 1.500/1.192 ms（实测
1.445/1.193 ms）。但同一点把 1T-only/after-68 预测为 1.254/1.171 ms（实测
0.907/0.921 ms），不能同时解释 domain-local latency-bound gather 与跨 domain
连续权重流；独立 M1 repeat 的五 context MAPE 为 17.34%，其中 1T-only/after-68
仍过估 38.03%/26.74%。故该式仅是默认关闭的可证伪原型，模型 schema/name 升为 9/v7 以隔离 cache，
**不生成或冻结新 calibration，也不进入 VND/LNS**。下一 probe 必须分别测 1T/16T
gather 的 phase duration、有效 DRAM/LLC-domain 流量和 victim W13 overlap；在此之前禁止
把 $\alpha_g$ 当成通用带宽倍率。

#### 9.44 LLC-domain memory-injection ceiling（默认关闭）

独立 placement probe 固定 target、aggressor 数量、route、权重与总工作，仅把十五个 1T
aggressor 从 target 所在 LLC domain 移到另一个 LLC domain。两次 31-round session 中，
local-minus-remote 的 head 差为 0.244/0.248 ms，after-1（target 开始时 15/15 peer 均在
W13）差为 0.225/0.233 ms；四个 paired P10 均大于 0.21 ms。locality penalty 在 gather
transition 与 W13 stream 两种窗口近似保持，因此首要缺口选择 domain injection，而不是
先拆两个独立资源。

令 LLC domain 集为 $\mathcal D$，rank DRAM service curve 为 $C_R(n)$，饱和值为
$C_R^{sat}$。可选 calibration $\beta>0$ 定义：

$$
C_{d,inj}(n_d)=\min\left(C_R(n_d),
\beta\frac{C_R^{sat}}{|\mathcal D|}\right),
$$

$$
R_{d,inj}=\sum_{i:d\cap CPU_i\ne\varnothing}
\frac{|CPU_i\cap d|}{a_i}\frac{Q_{i,dram}}{T_{i,dram}},
\qquad
D_{d,inj}=\max\left(1,\frac{R_{d,inj}}{C_{d,inj}}\right).
$$

其中 $a_i$ 是 phase active threads；跨 domain task 按 active-thread share 分流。task 的
DRAM dilation 取既有 rank dilation 与所触及 domain injection dilation 的最大值。event
explanation 同步保存每个 domain 的 active threads、offered rate、capacity、utilization 和
dilation。字段缺失时不计算 domain cap，旧 calibration 保持 rank-only 行为。

原型 $\beta=0.76$ 把 local-minus-remote 的 head/after 预测从约 0.004/0.004 ms 修正到
0.145/0.235 ms，mean absolute contrast error 从约 0.230 ms 降到 0.055 ms；但与冻结 v8
residual 直接叠加后，旧 25 点 route-context holdout MAPE 从 21.7% 恶化到约 28.5%。这说明
wide-team、narrow-team 与新 domain cap 对同一 contention 有双计数。故只接受资源结构，
不冻结 $\beta$，模型 identity 升为 schema 10/v8，VND/LNS 继续关闭；后续必须联合重估或
替换旧 residual，再以 route sweep 和三条 real trace 做 holdout。

#### 9.45 Phase re-accounting 与不可识别的 gather coupling

为避免 whole-expert total residual 掩盖 phase 误差，独立 fit corpus 使用与旧 holdout
不相交的 route 集 $\{3,4,7,8,10,16,24,48,68,600,1800\}$ 和全部支持宽度
$\{1,2,4,8,16,32,40,80\}$，对 gather/W13/W2 分别拟合：

$$
T_g(M,t)=\max\left(G_t, r_t\left\lceil\frac Mt\right\rceil\right),
$$

$$
T_s(M,t)=\max\left(F_{s,t},\gamma_{s,t}T_s^{physical}(M,t)\right),
\qquad s\in\{W13,W2\}.
$$

$G_t$ 是 gather 小工作服务下限，$r_t$ 是每 worker-row 时间；$F_{s,t}$ 是 cold stage
服务下限，$\gamma_{s,t}$ 只缩放原 physical stage。参数只消费 phase trace envelope，
whole-expert span 仅验证三 phase 求和，不进入 loss。旧 expert fixed/route overhead 的拟合项、
wide-team pressure 和 narrow-team correction 在 candidate 中归零；panel/range restart 与机器
service curve 保留。模型 identity 升为 schema 11/v9。

第二个独立 phase session 的 gather/W13/W2/total MAPE 为
16.13%/3.49%/4.70%/4.38%。纯 holdout 的 $M=\{1,2,5,6,12\}$ isolated W13/W2/total
MAPE 为 1.46%/9.18%/3.69%，证明重分账本身有效。

随后只用 cross-LLC after-1 contrast 拟合 $\beta=0.787$，validation residual 为
0.0086 ms；head contrast 留下稳定 $>0.02$ ms residual，且预测值落在fit paired区间外，
按预声明规则需要 gather/stream coupling。只用 head contrast 拟合得到 $\alpha_g=22.26$，fit/repeat residual 为近零/0.0040
ms。但这个值只由差分约束，local/remote 的巨大共模误差被抵消，不能识别 absolute pressure。

冻结后才读取的 25 点 route-context holdout 将 MAPE 从 frozen-v8 的 21.72% 恶化到
273.97%；48 个可分辨 mode pair 中出现 2 个 false dominance。三条 real trace 完整找回
13/16/12 个旧 measured state，high/median/uniformish Spearman 为 0.345/0.075/0.368，
high-skew top-8 漏 measured best；high-skew/median 的 43/9 个可分辨 pair 中分别出现
10/6 个 false dominance。因此接受 phase floor/scale 结构，拒绝完整 candidate 和
$\beta/\alpha_g$ 参数；不得替换 frozen v8 或接入 VND/LNS。下一 probe 必须同时提供
absolute slowdown 与 local-minus-remote contrast，并扫描 aggressor 数量，才能联合识别
domain cap 与 gather traffic。

#### 9.46 联合 absolute/contrast 拟合仍不能识别 $\beta$ 与 $\alpha_g$

本节不改公式、不升 schema、不替换 frozen v8。拟合器重放 9.45 已接受的
$T_g=\max(G_t,r_t\lceil M/t\rceil)$ 与 $T_s=\max(F_{s,t},\gamma_{s,t}T_s^{physical})$，
将 wide-team / narrow-team 与 whole-expert overhead 置 identity/zero，并在锁定的
aggressor-count probe 上联合最小化

$$
L=\mathrm{MAE}(\Delta^{\mathrm{abs}})+\mathrm{MAE}(\Delta^{\mathrm{loc}}),
$$

其中 $\Delta^{\mathrm{abs}}$ 是每个非 isolated mode 相对 matched isolated 的 median
span，$\Delta^{\mathrm{loc}}$ 是同一 count/phase 的 $\mathrm{same\text{-}LLC}-\mathrm{cross\text{-}LLC}$。
split 只进入绝对族。$\alpha_g$ 搜索上界固定为 8，禁止回到 contrast-only 的 22.26。
session 2 只做 validation；holdout SHA 在读入前拒绝。

Arm-codex NUMA3 两次 31-round count sweep（seed `20260907`/`20260908`）的硬件事实见
`optimizations/fused_moe_sve/results/arm_codex_80c_absolute_pressure_20260904.md`：
same-LLC 绝对减速在约 4 条 68-route 1T 流处饱和（head 相对 isolated 约
$+0.14/+0.20/+0.29/+0.30/+0.31\,\mathrm{ms}$，$n=1/2/4/8/15$），cross-LLC 相对
isolated 接近 0，split 看起来像 local 而不是 local/remote 平均。

session-1 粗网格 $\beta\in[0.20,2.00]$、$\alpha_g\in[0.25,8.00]$ 后邻域加密，最优点为
$\beta=0.78$、$\alpha_g=0.25$（贴 $\alpha_g$ 下界）。fit 绝对/对比/联合 MAE 为
$0.549/0.140/0.689\,\mathrm{ms}$；session-2 联合 MAE $0.719\,\mathrm{ms}$，对比残差全为正、
均值 $0.171\,\mathrm{ms}$，超过预声明的 $0.03\,\mathrm{ms}$ 同号门槛。近优集合有 739 个点
（loss $\le 1.05L^\star$），$\beta$ 跨度比 2.82、$\alpha_g$ 跨度比 11，判定不可辨识。
该点不是被拒绝的 contrast-only $(0.787,22.26)$。

即使 $\alpha_g$ 取下界，模型仍把高 count 的绝对压力高估约 3--5 倍：fit session 的
same-LLC head $n=15$ 实测 $+0.312\,\mathrm{ms}$、预测 $+1.057\,\mathrm{ms}$；cross-LLC
head $n=15$ 实测 $+0.043\,\mathrm{ms}$、预测 $+0.893\,\mathrm{ms}$。head $n\le 4$ 的
预测 contrast 为 0，因为 same 与 cross 被同一份 rank DRAM dilation 拖到几乎相同的
span。离线消融表明，关闭 domain cap 或把 $\alpha_g$ 从 0.25 提到 1.00，n15 head 的
绝对预测几乎不变（约 $+0.90\,\mathrm{ms}$）；$\beta=2$ 与 domain-off 重合。因此主导共模
来自既有 rank `dram_bytes` 曲线对重叠 68-route W13/W2 的共享服务，而不是 gather
offered-rate。把 $\beta$ 降到 0.20 能造出 locality contrast，但 same-LLC n15 head
绝对预测跳到 $+5.38\,\mathrm{ms}$。

仍不可识别的物理量是 **DRAM contention 的作用域与饱和形态**，不是再一个
$(\beta,\alpha_g)$ 数值：

1. 远程 LLC 上的 68-route 流为何几乎不通过 rank DRAM 拉长本地 1-route 1T victim，
   而模型用共享 $C_R(n)$ 对所有并发 task 做 dilation；
2. 本地 victim 为何在约 4 条 same-LLC 流处饱和，而 rank/domain utilization 会一直
   长到 $n=15$；
3. domain injection 只能叠在这份过大的共模之上，因此无法同时拟合 $\approx 0.3\,\mathrm{ms}$
   的 local plateau 和 $\approx 0$ 的 remote。

禁止为此加经验 residual、禁止扩大 $\alpha_g$ 上界、禁止读取 holdout、禁止替换
frozen v8 或接入 VND/LNS。下一步必须先改 DRAM 作用域（例如 domain-local DRAM
service，或 victim-asymmetric dilation），而不是重估 9.44/9.45 的两个标量。完整拟合
记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_absolute_pressure_joint_fit_20260904.md`。
锁定 DAG 上的 rank DRAM vs domain-only 消融见 9.47；rank LLC vs domain LLC
消融见 9.48。

#### 9.47 Rank DRAM 只解释约一半远程共模，不能单独改成 domain-only

本节不改公式、不升 schema、不替换 frozen v8。诊断脚本在与 9.46 相同的锁定
session-1 DAG 上比较预声明的作用域臂，不搜索 $(\beta,\alpha_g)$。关掉 rank DRAM
dilation 时，只把 placed allocator 中第一次 `service_rate(dram_bytes)`（rank 共享容量）
设为无穷，后续 domain injection 仍使用原来的 $C_R(n_d)$；禁止直接放大
`dram_bytes` 曲线，否则 equal-share cap 会一起被抬高。

预声明判定：远程 n15 head 绝对预测须 $\le 0.08\,\mathrm{ms}$，n4 locality contrast
须 $\ge 0.10\,\mathrm{ms}$，且 same-LLC n15 与 n4 之差 $\le 0.05\,\mathrm{ms}$。只有三
者同时被单一臂满足才允许新增 default-off 结构。

结果：rank-only 的 cross/same n15 head 为 $+0.893/+0.897\,\mathrm{ms}$。去掉 rank
DRAM 后降到 $+0.454/+0.590\,\mathrm{ms}$，远程共模只去掉约 $49\%$。剩余 dilation 由
`llc_bytes` 主导（cross n15 的 LLC dilation $\approx 2.44$，L2 仅 $\approx 1.10$），
gather 与首个 W13 cold 面板仍不膨胀。domain-only $\beta=0.78$ 把 n4 contrast 抬到
$0.543\,\mathrm{ms}$，但远程仍是 $+0.454\,\mathrm{ms}$，same-LLC n15$-$n4 仍为
$0.328\,\mathrm{ms}$；$\beta=0.20$ 把 same n15 推到 $+5.71\,\mathrm{ms}$。没有任何臂
达到预声明的单一资源门槛。

另外，n=1 的本地 $+0.144\,\mathrm{ms}$ 也不能由该消融恢复：rank DRAM 给 same 与
cross 同样的 $+0.082\,\mathrm{ms}$，去掉后 same n1 预测为 0。硬件是 saturating、
same-LLC-only 的小税；模型是随 count 增长的 rank/LLC utilization。

因此不新增 default-off 结构。锁定 DAG 上的 rank LLC vs domain LLC 消融见 9.48。

#### 9.48 Rank LLC 解释全部剩余远程共模，domain LLC 仍是随 count 增长的本地税

本节不改公式、不升 schema、不替换 frozen v8。诊断脚本在与 9.46/9.47 相同的锁定
session-1 DAG 上比较预声明的 LLC 作用域臂；全部臂保持 rank DRAM 关闭、
$\alpha_g=1$、domain DRAM injection 关闭，不搜索参数。

placed allocator 对 LLC 走两条容量：逐域单 key 的 $B_d(n_d)$，以及传入全部
domain id 的 rank 调用。单活跃域时 rank 容量等于 domain 容量；两域同时活跃时
rank 取 $\min(\sum B_d, B_{LLC,rank}^{sat})$。task 的 LLC dilation 为
$\max(\mathrm{rank},\mathrm{local})$。关掉 rank LLC 时，只对
`len(llc_domain_threads)>1` 的 `service_rate(llc_bytes)` 返回无穷；禁止套用
DRAM 的第一次调用规则，因为 domain LLC 先于 rank LLC 被调用。

预声明判定与 9.47 相同：远程 n15 head 绝对预测须 $\le 0.08\,\mathrm{ms}$，n4
locality contrast 须 $\ge 0.10\,\mathrm{ms}$，且 same-LLC n15 与 n4 之差
$\le 0.05\,\mathrm{ms}$。只有三者同时被单一臂满足才允许新增 default-off 结构。

结果：`no_rank_dram` 的 cross/same n15 head 为 $+0.454/+0.590\,\mathrm{ms}$。
`domain_llc_only` 把远程打到 $0$，本地曲线不变，因此
`rank_llc_fraction_of_leftover_remote_n15=1$。same-LLC 下两臂逐点相同，符合
单活跃域时 rank LLC 已等于 domain LLC。事件上，cross n15 的 victim domain LLC
dilation 已是 $1.00$，aggressor domain 为 $2.84$，victim 通过 rank fabric
$2.44$ 继承远程共模；关掉 rank LLC 后 victim phase dilation 变为 $1$。

但 `domain_llc_only` 的 same-LLC n15$-$n4 仍为 $0.194\,\mathrm{ms}$，count 形状
为 $0.168/0.397/0.447/0.590\,\mathrm{ms}$（n2/4/8/15），达不到 n≈4 饱和。
`no_llc` 把本地 n15 降到 $+0.001\,\mathrm{ms}$，`no_llc_no_l2` 与之相同，L2
dilation $\approx 1.10$ 不是剩余项。去掉全部共享 cache/DRAM 后，模型中的
1-route 1T victim 近似 isolated，而硬件 same-LLC n15 仍为 $+0.312\,\mathrm{ms}$、
n1 为 $+0.144\,\mathrm{ms}$。

因此不新增 default-off 结构。锁定 DAG 上的 victim-asymmetric dilation 消融见 9.49。

#### 9.49 1-route victim 继承的是 cohort 字节 dilation，own-demand 会把它打成 isolated

本节不改公式、不升 schema、不替换 frozen v8。诊断脚本在与 9.46--9.48 相同的锁定
session-1 DAG 上比较 dilation **分配规则**，不搜索参数、不加 additive residual。
全部臂保持 domain DRAM injection 关闭，并保留 rank DRAM 与 rank LLC 容量。

对称规则把 cohort 的 DRAM/L2/LLC dilation 套到每一个并发 task。对照臂为：
孤立 GEMM$\ge$transfer 的 phase 不再吃 transfer dilation（`compute_bound_skip`）；
只把同一 LLC domain 内 task 的 DRAM offered 计入 victim，并去掉 rank LLC
（`same_llc_peers`）；每个 task 只用自己的 offered/capacity
（`own_demand`）。禁止把 peer count 截断到 4 作为可接受结构，因为那会在定义
gate 的同一条 count 曲线上**强制**制造平台。

孤立瓶颈：1-route 1T 的 W13 cold 是 transfer-bound（DRAM $0.243\,\mathrm{ms}$，
GEMM $0.189\,\mathrm{ms}$），68-route 1T 是 GEMM-bound（$1.134$ vs $0.243$）。
两者 cold packed-B 几乎相同（约 $8.4\,\mathrm{MiB}$）。因此
`compute_bound_skip` 与对称臂逐点相同。

结果（head isolated-relative）：对称臂 same/cross n15 为
$+0.897/+0.893\,\mathrm{ms}$。`same_llc_peers` 把远程打到 $0$，本地曲线不变。
`own_demand` 把本地/远程 n15 打到 $+0.001/0\,\mathrm{ms}$（单条 cold-B 流约
$24\,\mathrm{GB/s}$，低于 rank DRAM $166\,\mathrm{GB/s}$ 与 domain LLC
$216\,\mathrm{GB/s}$）。没有任何臂同时满足远程近零、n4 contrast 与 n≈4 平台。

因此不新增 default-off 结构。own-demand 证明过预测来自「短 transfer-bound
victim 继承 68-route cohort 字节利用率」；硬件留下的是
$\approx 0.31\,\mathrm{ms}$、n≈4 饱和、same-LLC-only 的 occupancy 税，不能由现有
service curve 识别，也不能在本 count sweep 上拟合 additive residual。固定 count
变 aggressor $M$ 的识别见 9.50。holdout 仍未读。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_victim_asymmetric_dilation_20260904.md`。

#### 9.50 固定 count 变 aggressor $M$，识别 leftover 是 occupancy 还是利用率

本节不改公式、不升 schema、不替换 frozen v8、不读 holdout。识别必须独立于
n=1/2/4/8/15 count 曲线：victim 仍是 1-route 1T，peer 人数固定在孤立对照、n=1
与 n=4，aggressor 的 route 数取 $M\in\{1,4,16,68\}$。packed-B 几乎不随 $M$ 变
（约 $8.4\,\mathrm{MiB}$）；A/C 流与 GEMM 时长随 $M$ 增长。$M=1$ 的 aggressor
与 victim 一样是 transfer-bound，$M=68$ 是 GEMM-bound。

预声明判定（head isolated-relative），本 probe **不加结构**：

- occupancy：same-LLC n1 与 n4 从 $M=1$ 到 $M=68$ 的变化绝对值 $\le 0.05\,\mathrm{ms}$
- duration occupancy：$M=1$ 的 n4 税 $\le 0.08\,\mathrm{ms}$，且 $M=16$ 与 $M=68$
  相差 $\le 0.05\,\mathrm{ms}$，同时 $M=68$ 的 n4 税 $\ge 0.10\,\mathrm{ms}$
- utilization：n4 税从 $M=1$ 到 $M=68$、以及从 $M=16$ 到 $M=68$，增长都
  $\ge 0.15\,\mathrm{ms}$
- overlap：same-LLC n1/n4 的 `peer_overlap_experts` 中位数至少为 $0.8\times$ count，
  否则本 cell 作废

probe：`optimizations/fused_moe_sve/benchmarks/bench_aggressor_m_occupancy.py`。
Affinity 与 count sweep 相同：`taskset -c 240-319 numactl --membind=3`，5 warmup、
31 randomized rounds、4 measured copies、每 sample 前用最大 $M$ 的 isolated scrub。
$M$ 与 placement 在同一轮内打乱，避免跨 $M$ 时间漂移。未启用的 background 仍
dependency-delay 到 target lane tail，每个 $M$ 内部保持同一 route histogram。

Session-1（seed `20260909`，SHA
`f1daf1e5b4aa02cc9d0c61fcfbde72005eb4f678be2bb20e48d25e031a3217b0`）overlap 有效，
远程 n4 近零。same-LLC n4 head isolated-relative 为
$M=1/4/16/68$：$+0.660/+0.639/+0.402/+0.290\,\mathrm{ms}$。$M=68$ 的 $+0.290$ 与
锁定 count sweep 的 n4 $+0.291$ 一致。P10/P90 把 $M=1$（$0.637$--$0.701$）与
$M=68$（$0.274$--$0.317$）分开。预声明三态都不成立：税随 $M$ **下降**，不是
occupancy 平台，也不是 utilization 上升。after_1 上 $M=1/4$ 税消失，因为短
peer 在 1-route delay 期间已经结束，不是 head 的对照。

因此不新增结构。leftover 是同 LLC 上并发 **transfer-bound** 流的争用：短
DRAM-bound peer 罚得更重，长 GEMM-bound peer 反而更轻。holdout 仍未读。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_aggressor_m_occupancy_20260904.md`。

#### 9.51 同一 16 核上拆 1×16T 与 16×1T

本节不改公式。问 16 线程宽队是一条访存流还是 16 个填数口。victim 仍是 1-route
1T；全部 aggressor $M=1$（transfer-bound）。对照核为 LLC7 logical `48-63`
（不含 victim 64）。预声明：`fill_ports` 若 1×16T 与 16×1T 的 same-LLC 税相差
$\le 0.10\,\mathrm{ms}$ 且 16×1T $\ge 0.40\,\mathrm{ms}$；`one_stream` 若 1×16T
接近单条 1T（$\le 0.10\,\mathrm{ms}$）且 16×1T 比它至少再重 $0.20\,\mathrm{ms}$。
Arm session-1 overlap 有效、远程近零；same-LLC head 为 1T / 1×16T / 16×1T 的
$+0.018/+0.014/+0.160\,\mathrm{ms}$。`fill_ports` 不成立。`one_stream` 方向成立
但 many$-$wide 仅 $0.146\,\mathrm{ms}$，未过 $0.20\,\mathrm{ms}$ 门槛，故
`inconclusive`。1×16T $M=1$ 是一条 packed-B，不是 16 条独立填数。本 probe
不加结构。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_fill_port_vs_stream_20260904.md`。

#### 9.52 数流不数线程：`8+8+4+1` 对照

本节不改公式。问 leftover 税跟并发 packed-B 流数走，还是跟线程数走。victim
仍是 1-route 1T。全部 aggressor $M=1$。等线程阶梯固定 16 线程于 logical
`48-63`，流数为 $1/2/4/16$（1×16T、2×8T、4×4T、16×1T），另用 4×1T 作四流对照。
用户构图 `8+8+4+1` 放在 `43-63`（21 线程、4 流），对照同起点 4×1T 与 21×1T。
预声明：`stream_count` 若 4×4T 更接近 4×1T 而不是 16×1T，且 `8+8+4+1` 更接近
四流 1T 而不是 21×1T；`thread_count` 则相反。Arm session-1 overlap 有效；
等线程阶梯 same-LLC head 为 1×16T / 2×8T / 4×4T / 4×1T / 16×1T 的
$+0.003/+0.035/+0.075/+0.070/+0.155\,\mathrm{ms}$；`8+8+4+1` / 四起点 1T /
21×1T 为 $+0.110/+0.103/+0.301\,\mathrm{ms}$。自动化签名 `stream_count`。
21×1T 远程 $+0.211\,\mathrm{ms}$，四流 mix 远程 $+0.042$。本 probe 不加结构。
完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_count_composition_20260904.md`。

#### 9.53 整层权重放进一块连续 DRAM

本节不改公式。当前 packed 已是每矩阵一块 `[E,\mathrm{packed}]`；kernel 要求
contiguous 2-D，不能在专家之间插入空洞。本 probe 把一层 W13 与 W2 再拷进
同一匿名 allocation 的两个 view，对照现在的两块 tensor。问的是：再并成一块
会不会降低 16×1T leftover。它不能把 16 条并发专家流变成 1 条流。victim 仍是
1-route 1T，全部 $M=1$，核位与 fill-port 相同。本机未预留 HugeTLB。预声明：
`unified_helps` 若 unified 比 split 至少轻 $0.08\,\mathrm{ms}$；`layout_neutral`
若两者相差 $\le 0.04\,\mathrm{ms}$。Arm session-1 overlap 有效；16×1T leftover
split/unified 为 $+0.153/+0.157\,\mathrm{ms}$，差 $0.004$，签名 `layout_neutral`。
1×16T 为 $+0.041/+0.042$。远程 16×1T 为 $+0.088/+0.098$，略超 $0.08$ 报告门槛。
本 probe 不加结构。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_unified_weight_block_20260904.md`。

#### 9.54 权重块 THP 对照 4 KiB

本节不改公式。unified-block 已证明把 W13+W2 拼成一块不降 leftover，但
`torch.empty` 可能静默继承 THP。本 probe 把同一块 W13+W2 拷进
`mmap+MADV_NOHUGEPAGE` 与 `mmap+MADV_HUGEPAGE`，用 `/proc/self/smaps` 的
`AnonHugePages` 验证页大小。victim 仍是 1-route 1T，全部 $M=1$，核位与
fill-port 相同。不改 `FUSED_CPP_PAGES`，scratch 保持进程默认。本机 THP 为
`[always]`，未预留 HugeTLB。预声明：`thp_helps` 若 4 KiB 比 THP 至少重
$0.08\,\mathrm{ms}$；`page_neutral` 若两者 leftover 相差 $\le 0.04\,\mathrm{ms}$；
`thp_not_latched` 若 4 KiB 仍有大页或 THP 覆盖不足 $80\%$。Arm session-1 overlap
有效且页验证通过：4 KiB AnonHugePages $=0$，THP 覆盖 $100\%$。16×1T leftover
small/THP 为 $+0.167/+0.162\,\mathrm{ms}$，差 $0.005$，签名 `page_neutral`。
1×16T 为 $+0.006/+0.008$。isolated 绝对 span 为 $0.577/0.571\,\mathrm{ms}$。
远程 16×1T 为 $+0.145/+0.143$，高于 $0.08$ 报告门槛。本 probe 不加结构。完整
记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_weight_thp_20260904.md`。

#### 9.55 core PMU、L3C 与 DDRC 对 leftover 的硬件分账

本节不改公式、不读 holdout。PMU 不能直接包住 9.52 的完整等工作量 DAG，因为被
禁用的 background 会在 victim 结束后执行并污染 uncore 总量。因此新增 Lab-only
`bench_stream_pressure_pmu.py`：每个 cell 只保留 victim 与实际并发 peer；初始化、
warmup 和第五份不相交 packed copy 的 18-expert scrub 均在 counter disabled 时完成。
两个嵌套 `perf stat --delay=-1 --control=fifo` 在同一次 31-run cell 上同步采 CPU304
core PMU、LLC7 十个 L3C slice、NUMA3 八个 DDRC 与 native target trace。

session-2/3 的 victim span 在 isolated、1x16T、4x4T、4x1T、16x1T 上分别为
$0.482/0.519/0.630/0.691/0.705$ 与
$0.477/0.517/0.627/0.644/0.704\,\mathrm{ms}$；除 4x1T 外跨 session 差不超过
$0.005\,\mathrm{ms}$。session-3 的 DDR read-command latency 为
$32.6/43.9/55.8/49.2/57.0$ cycles，victim LLC read-miss ratio 为
$5.3/13.3/33.8/55.9/52.7\%$，backend-stall ratio 为
$70.6/71.3/74.3/75.7/75.6\%$。这些延迟/失速量在约四份互异 B 后趋于平台；相反，
DRAM read traffic 仍为 $10.0/21.5/37.0/37.5/108.3\,\mathrm{MiB/call}$，估算带宽
为 $5.8/12.1/19.0/19.0/42.0\,\mathrm{GB/s}$，没有在四流处平台。

因此排除“aggregate DRAM bytes 或带宽本身直接线性决定 victim leftover”。当前
物理解释是：互异 transfer-bound packed-B fill 增加 victim-visible LLC miss 与
DDRC read-command 排队，二者对应硬件 leftover 的饱和形状。但 4x4T 与 4x1T 的 DRAM
量相同、victim miss ratio 不同，且 4x1T span 跨 session 漂移 $0.048\,\mathrm{ms}$；
stream count 不是完整定量变量。本节只完成来源分账，不新增结构或经验 residual。
若继续，必须独立 sweep $0/1/2/4/8/16$ 互异 transfer-bound B、重复 4x1T，并以
leave-one-count-out 和独立 session 识别 queue/miss pressure 参数。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_pmu_20260904.md`。

#### 9.56 packed-B count sweep 的 queue/miss LOCO 比较

本节不改公式、不读旧 route/real-trace holdout。PMU-only probe 增加 2x8T 与 8x2T，
在固定 16 个 peer 线程下 sweep $0/1/2/4/8/16$ 份互异 transfer-bound packed-B；
另以独立 seed/session 重复 isolated 与 4x1T。core/L3C/DDRC 仍由两个 perf FIFO 在
同一次 cell 同步采集，5 warmup、31 measured、4-copy rotation 与 18-expert scrub
协议不变。全部 event running 为 $100\%$，overlap 中位数精确为
$0/1/2/4/8/16$。

主 session 的 count/spans 为
$0/1/2/4/8/16:\;0.5690/0.5410/0.5811/0.6000/0.6787/0.6774\,\mathrm{ms}$；
DDRC queue latency 为 $36.46/45.36/49.96/57.38/65.02/57.81$ cycles；victim LLC
read-miss ratio 为 $0.3656/0.2679/0.3705/0.3853/0.4883/0.4331$。count-1 的负
span delta 与负 LLC pressure 均保留在主分析中；它的非配对 P10--P90 与 isolated
重叠，不能声称稳定收益。

主比较使用过 isolated 原点、非负 slope 的单变量 LOCO。DDRC queue/LLC miss 的
MAE 为 $0.0453/0.0364\,\mathrm{ms}$，RMSE 为 $0.0464/0.0459\,\mathrm{ms}$，最大
绝对误差为 $0.0574/0.0767\,\mathrm{ms}$。LLC 虽有较低 MAE，但 count-1 pressure
为负并产生一个非正预测，不满足单调压力变量；queue 在所有 count 保持非负，但
低估 count 8/16。允许 intercept 的敏感性 LOCO 也只把 MAE 降到
$0.0379/0.0274\,\mathrm{ms}$，不改变不可冻结结论。

固定 16-thread ladder 之外的 4x1T layout holdout 上，主/独立 repeat 的 queue 误差
为 $-0.0274/-0.0010\,\mathrm{ms}$，LLC 误差为
$+0.0558/+0.0248\,\mathrm{ms}$。因此 queue latency 只作为下一轮 measurement-design
候选，LLC miss ratio 只作解释与 guardrail；两者均不进入 cost model。当前主要
identifiability 缺口是每个 mode 在独立进程中初始化，isolated 绝对 span 可跨 session
漂移。下一步若继续，应在同一长生命周期 allocation/process 中配对 isolated 与
candidate，并获得 per-cell counter reset/read，而不是增加回归参数。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_count_loco_20260904.md`。

#### 9.57 同进程、同 allocation、逐 cell counter reset 的 paired PMU

本节不改公式、不读旧 holdout。9.56 的 mode-per-process 协议仍有 isolated baseline
漂移；本 probe 在一个长生命周期进程中一次性构建全部 plan/input/output、四份轮换
measured packed allocation 与第五份 scrub allocation。每轮将 isolated、1x16T、
2x8T、4x4T、8x2T、16x1T、4x1T 各执行一次并随机排序。Linux
`perf_event_open` 一次打开 60 个 CPU304 core、LLC7 L3C 和 NUMA3 DDRC event；每个
cell 独立 reset/enable/run/disable/read。warmup 后控制线程固定 CPU240，victim 保持
CPU304，避免 Python control syscall 进入 victim core 计数。

主 session（31 paired rounds）全部 event 的 running ratio 为 1，overlap 中位数精确
为 $0/1/2/4/8/16$。count $0/1/2/4/8/16$ 的绝对 victim span 为
$0.5833/0.5909/0.6120/0.6211/0.7821/0.7798\,\mathrm{ms}$；同轮 paired delta 为
$0/+0.0082/+0.0295/+0.0455/+0.2040/+0.2042\,\mathrm{ms}$。count 8/16 的 delta
P10 为 $+0.1616/+0.1466\,\mathrm{ms}$，形成稳定平台；count 1 区间跨零，count 4
P10 为 $-0.0004\,\mathrm{ms}$。DDRC queue pressure paired median 为
$0/+11.1/+19.0/+21.0/+41.8/+36.0$ cycles，所有正 count 的 P10 均为正，也在 4 到
8 之间跳变。victim LLC miss pressure 在 count 1/2/4 为负，且除 count 8 外 P10
多跨零，不能作为稳定单调压力变量。

isolated-anchored queue/LLC LOCO MAE 为 $0.0515/0.0778\,\mathrm{ms}$，RMSE 为
$0.0528/0.0860\,\mathrm{ms}$，最大误差为 $0.0677/0.1220\,\mathrm{ms}$；LLC 有三个
非正预测。允许 intercept 的敏感性仍由 queue 胜出，MAE 为
$0.0355/0.0432\,\mathrm{ms}$。独立 4x1T paired repeat 测得 slowdown P10/median/P90
$=+0.0329/+0.0878/+0.1374\,\mathrm{ms}$、queue pressure median $+13.50$ cycles；
queue 线性模型稳定低估约 $0.028\,\mathrm{ms}$，LLC 则高估 $0.071\,\mathrm{ms}$。

因此接受 paired per-cell PMU 协议，拒绝 LLC standalone feature；queue latency 是
明确胜出的物理变量，但当前线性形式仍在 count 1--4 过估、count 8--16 低估，不能
冻结。不得从本 session 添加 count knee、queue threshold、双特征拟合或 layout
residual。若继续，应固定互异 B count，独立改变 request-arrival/team shape，寻找
planner 可见的 queue-pressure proxy；production planner 不能读取未来 PMU。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_paired_pmu_20260904.md`。

#### 9.58 固定 packed-B count 的 request-arrival shape 分解

本节不改公式、不读旧 holdout。在 9.57 的同进程逐 cell PMU 协议上固定 expert id、
$M=1$、packed allocation、same-LLC placement 与 expert 起点，只改变每份 B 的 team
width：count 4 在 logical $48/52/56/60$ 对照 4x4T/4x2T/4x1T；count 8 在
$48/50/\ldots/62$ 对照 8x2T/8x1T。两次独立 session 均为 5 warmup、31 randomized
paired rounds，60 个 event 全部 running ratio 1，overlap 等于固定 B count。

count 4 下，4x4T/4x2T/4x1T 相对 isolated 的 queue pressure 在 session-1 为
$+18.82/+17.06/+13.44$ cycles，session-2 为 $+20.94/+16.13/+11.12$ cycles；team
变窄稳定降低 aggregate queue。但 4x1T-4x4T 的 victim span 为
$-0.0338\,[-0.0618,+0.0026]$ 与 $+0.0225\,[-0.0122,+0.0512]\,\mathrm{ms}$，
两轮方向不同且区间跨零；4x2T-4x4T 也在两轮跨零。故 count 4 未识别出稳定 width
对 victim latency 的影响。

count 8 下结果稳定：8x1T-8x2T 的 victim span 在两轮为
$-0.1133\,[-0.1676,-0.0815]$ 与 $-0.0775\,[-0.1411,-0.0247]\,\mathrm{ms}$；
queue latency 为 $-16.90\,[-30.68,-7.65]$ 与
$-14.06\,[-20.61,-7.19]$ cycles。较窄 team 同时稳定降低 queue 与 victim latency。
预声明的“narrower slower despite lower queue”在全部 pair 上拒绝；count 8 接受
`narrower_faster_with_lower_queue`。

因此互异 transfer-bound B count 是必要但不充分的 pressure 变量。team width 不增加
互异 B，却增加同一 B 上并发 N-stripe requester 与短窗 injection rate；硬件压力至少
依赖 $(n_B,\{t_e\})$ 的交互。相同总 peer 线程下，8x1T 比 4x2T queue 更高；相同
$n_B=8$ 下，8x2T 比 8x1T queue 更高。count 4 仍在 latency 噪声区，count 8 进入可
分辨高压区。只有两个 count，禁止拟合 count knee、width multiplier 或 residual。
若继续，应以 start-aligned count $4/6/8$ × width $1/2$ 的独立网格验证 planner-visible
injection proxy，并记录 per-stage overlap duration。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_request_shape_20260904.md`。

#### 9.59 count-6 锁定 holdout 的结构 proxy 停止门槛

本节不改公式、不读旧 route/real-trace holdout。以 nested-prefix 起点构造 count
$\{4,6,8\}\times$ width $\{1,2\}$ 网格；只用 count 4/8 拟合，count 6 在运行前锁为
holdout。两次独立同进程 paired session 均为 31 rounds、逐 cell 60-event PMU；全部
running ratio 为 1。候选结构 proxy 预声明为 $n_B$、active requester threads、二者
乘积，以及 $n_B\times$ measured W13 overlap core-ms oracle。每个 proxy 先过原点预测
DDRC queue，再由同一训练集 queue slope 预测 victim slowdown。

预声明门槛要求两个 count-6 width 在两轮同时满足：queue error
$\le\max(3\text{ cycles},15\%)$、slowdown error $\le0.02\,\mathrm{ms}$、1T/2T
方向正确，并且 proxy-to-queue 与 queue-to-slowdown 参数跨 session 漂移 $\le20\%$。
count-6 实测在两轮为：1T queue $16.93/16.29$ cycles、slowdown
$0.0685/0.0650\,\mathrm{ms}$；2T queue $22.50/18.45$ cycles、slowdown
$0.1166/0.1103\,\mathrm{ms}$。holdout 自身足够稳定用于拒绝。

除只看 $n_B$ 因无法区分 width 而方向失败外，其余 proxy 方向正确，但全部失败。
$n_B$/requesters/product/oracle 的
proxy-to-queue slope 漂移为 $26.2/28.7/32.3/25.8\%$。oracle 仅 session-1 通过；
session-2 的 1T queue error 为 $-5.60$ cycles，2T slowdown error 为
$-0.0268\,\mathrm{ms}$。相反，queue-to-slowdown slope 为
$0.00440/0.00476\,\mathrm{ms/cycle}$，只漂移 $7.5\%$。因此不可辨识点位于
plan geometry 到 memory-controller queue state 的映射，而不是 victim slowdown
系数。

最终 `accepted=[]`、`stop_absolute_model_expansion=true`。即使 measured-overlap oracle
也不能跨 session 通过，禁止继续添加 count knee、width multiplier、queue threshold、
双特征项或经验 residual。绝对基线保持 frozen v8；后续安全性转向 anchor-relative
partial order、top-K recall 与 false-pruning replay，只有这些门槛通过后才接 VND/LNS。
完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_proxy_grid_20260904.md`。

#### 9.60 事后 planner-visible geometry 不能预测 queue

在同一锁定网格上，不用实测 overlap，只使用 live team 与冻结 v8 的 isolated
W13/operator 相位，补测 \(n_B\sqrt{t}\)、predicted peer W13 core-ms、predicted
operator core-ms，以及过原点的 \(a n_B+b n_{\text{threads}}\)。门槛与 9.59 相同。
全部失败：proxy-to-queue 漂移 \(26.9\%\)--\(62.4\%\)。session 2 的 operator 与双系数
拟合可以单独过 count-6，但不能跨 session。同一 8x2T live-team 的 queue 为
\(48.168/29.142\) cycle（比 \(1.653\)），确定性 \(g(P)\) 无法给出稳定斜率。
`accepted=[]`，保持 `stop_absolute_model_expansion` 与 frozen v8。完整记录见
`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_plan_geometry_proxies_20260905.md`。

### 核数压力的完整 MoE 外推回放（Lab，2026-09-09）

`optimizations/fused_moe_sve/benchmarks/replay_core_pressure_moe.py` 提供独立
Lab 子类，不注册到 production planner，也不修改 frozen v8 calibration。
目的仅是检查当前核数压力响应接入完整 phase DAG 后的总耗时误差。

在每个 active phase 集合中，仅将 GEMM (`cold_b`/`steady_b`) 的 active CPU
计入局部竞争。对任务 i，排除自身，在其覆盖的各 LLC domain 中取最大 peer
core 数 n_i，令 x_i=n_i/32，f_i=1+0.28902964424300354*x_i+
0.4536242509371087*x_i^2。系数来自独立 M12 团队实验的 1T 训练条件，
本回放不拟合参数、不利用完整 forward 实测时间构造输入。

保留原模型的 domain spill fraction、global resource dilation、任务依赖和
isolated phase 成本；GEMM 的 LLC resource scale 替换为 f_i，DRAM scale
取 max(global DRAM dilation,f_i)。计算/访存仍以 max 合成，不将整个 GEMM
时间乘以 f_i。原 concurrent wide/narrow team correction 被替换，保留
isolated wide-team scale；只有一个 active phase 时直接沿用原模型。
非 GEMM phase 保持原 placed-state 结果。该接法的改善不能唯一归因于核数
指标，因为它同时替换了旧局部响应组合。

这是明确的外推假设：独立实验只覆盖 M1 W13 victim、M12 1/2/4T 背景及
steady-state 协议；完整回放覆盖 M1–1945、width1–16、W13/W2 和不同缓存
生命周期。不是已识别的 LLC 服务率，也不是 production adoption。
未更改剪枝、候选集合、Plan V2、native planner parity 或默认选择。

46 个历史样本位置、两场各方案中位数共 92 个观测：原 v8 MAE3.733ms /
MAPE12.018%，Lab 外推 MAE1.143ms / MAPE3.652%。uniformish 从 MAE1.166ms
退化到1.718ms；绝对时间改善不保证排序改善。样本已被历史筛选，跨组存在
相同方案，不是92个独立 holdout。阶段 trace 关闭，无法由这些数据给出实测
阶段误差分解。完整证据见 `optimizations/fused_moe_sve/results/core_pressure_full_moe_20260909.md`。

### 完整块与历史相关尾块拆分（Lab，2026-09-09）

独立实验 `full_tail_cost.py` 对 W13 K4096/N1024、1T 定义
`M=12q+r`，`T=q F(p)+R(r,q,p)`；`r=0` 时尾块成本为零。
`p` 为同 LLC 的独立 M12 后台任务数，端点0/38；F来自第一场 M192
完整块中位数，R按余数1/5/11、此前完整 B 扫描次数0/1/2/8/15分别采集，
在扫描次数和压力上作分段线性插值。M不超过192；未测余数显式拒绝。
不施加额外压力倍率，也不强制尾块由冷到热单调变化。

扫描次数4留出；第二场尾块 MAE18.04→3.71us，但留出位置的完整时间
MAE14.10→16.49us。真实独立 M13/17/35/65均改善；p11 M13联合实测
2.897ms仍超过模型0–38背景的敏感性范围1.564–1.940ms。该范围不是物理上界。
这是已查看数据的回顾性验证，未验证中间压力尾块、混合竞争者、W2或多线程。
不接入生产校准、默认调度或剪枝。详见
`optimizations/fused_moe_sve/results/full_tail_cost_20260909.md`。

### 新旧 cost model 的等预算 full/strict 搜索对照（Lab，2026-09-09）

在三条既有 high-skew/median/uniformish route 上分别从 route counts 重新执行
`IntervalPlanner.plan(dynamic_tail_pool=False, bounded_tail_repartition=False)`，
而非重排历史46个计划。保持现有141个模板、isolated-LPT、奇偶 lane reversal
和 `_select_analytic_full` 的不确定性 fallback；仅将模型替换为上文核数压力
Lab 子类。两边显式 `early_merge=False`，不改变 production 默认行为。

预算按完整模板集合及实际 `_score` DAG 调用核对，包含 quick winner 的额外
event 重评分；同时保存有序 task digest，不能仅以相同 iteration 数声称等预算。
这是现有 strict/fixed/full-stripe 空间的完整搜索，不含动态 tail pool、route
slicing 或额外 VND/LNS 扩展。原 residual comparator 不迁移为新模型的剪枝依据。
每条 route、每个模型以独立进程重复两次冷搜索，检查预算、计划与分数的确定性，
单独报告初始化和搜索耗时。选出的计划在任何新硬件结果可见前冻结。

沿用关闭 early merge 的 last-W2 计算完成时间口径；对共同 reference、旧模型
选择和新模型选择做同进程 paired 硬件对照及独立重复。先验证输出与实际 merge
状态，再报告计算时间、E2E、选择差距和搜索成本。此记录是 Lab 方法声明；结果
见 `optimizations/fused_moe_sve/results/full_model_search_compare_20260909.md`，
不授权生产模型导出、native planner parity 或超出既有 profile 的精度结论。

对照已完成：12 次正式搜索均为141模板/422次评分，计划、分数与有序 DAG
在独立重复中一致；7个去重计划的硬件对照全部通过。新选择在 high-skew
只改善约0.26%，median 改善1.96–2.03%，uniformish 选择不变；冷搜索成本
约为旧模型的1.85–1.91倍。更大的限制来自既有 fallback：high-skew/median
的低均值16T候选被替换为max-width8T计划，已测reference仍快约4.3–10.2ms。
本次不修改该规则，也不把旧不确定性校准视为已适用于新模型；后续选择器
验证必须与本次冻结对照分开。

### 固定评分池的 fallback 对照（Lab，2026-09-09）

固定上轮新核数压力模型的三条 route、141模板和422次评分，重建每个有序
DAG 并逐项匹配冻结 digest 后复用其分数；完整 ranking 和原 fallback bridge
必须与原搜索一致。对同一候选池比较既有 `_select_analytic_full` 与直接取
`(makespan, pessimistic, working_set, resource_groups)` 字典序最小候选。
只改变最终选择规则，不改变模型、候选生成、预算或 production 行为。
此操作是冻结池重放，不是重新计时的冷搜索；冷搜索成本沿用上轮证据。

两边 `early_merge=False`、full owner stripes，冻结 bridge 后在同一进程中
配对测量 last-W2 计算完成时间，两次独立 session 和无 trace E2E 对照。
uniformish 若两边 bridge 相同则去重，不能把同一计划解释为独立零方差样本。
结果记录于 `optimizations/fused_moe_sve/results/fallback_compare_20260909.md`；
不重新拟合不确定性，也不把三条既有 route 当作全域 holdout。

2026-09-10 验证完成：直接 mean-best 在 high-skew 两轮均降低计算时间约22.93%，
median 降低16.18%/12.55%；无 trace E2E 分别降低22.12%/15.52%。uniformish
计划相同。9次硬件运行正确，406个 trace call 无 early merge，6份压缩 trace
哈希一致。median 候选跨 session 漂移1.22ms，保留收益范围而不合并为固定值。
此结果支持后续 Lab 选择器决策；本次只验证，未修改既有 fallback 或生产默认。

2026-09-10 后续用户决策：实验基线采用“冻结新模型 + 直接预测最小值”。
Lab `bench_full_model_search.py` CLI 默认 `--model new --selector mean`，
`--selector fallback` 保留旧选择规则作为同模型对照。均值相同时仍依次按
pessimistic、working set、resource groups 打破平局。候选空间、模型系数、
early merge 关闭和 last-W2 时间口径不变；生产 Python/native planner 未迁移。
归档 helper 的默认 fallback 保持兼容，CLI 显式传入新默认，历史 runner 快照保留。

### 统一 mean-best 的新旧模型效果对照（Lab，2026-09-10）

两模型均使用 mean-best 选择，重建各自既有141模板/422次有序 DAG 的完整评分池，
保持模型、候选生成和 early merge 关闭设置不变。对各自选择计划的并集去重，
两模型都预测同一组计划，以该共同样本计算 MAE/MAPE/bias；另行比较所选计划的
实测计算时间，不能用不同计划上的各自误差替代共同样本精度比较。

三条 route 产生四个去重计划。median/uniformish 两模型选中完全相同的 bridge，
复用刚完成的对应实测；high-skew 的旧模型 reverse-odd 和新模型 LPT 在相同
`4×16T+16×1T` 下不同，补做两次31轮配对 trace 与无 trace E2E 对照。报告复用
来源、共同计划的误差和选择差距；不把三条既有 route 视作全域 holdout，也不将
之前去掉 fallback 的收益归因于模型替换。结果见
`optimizations/fused_moe_sve/results/mean_model_compare_20260910.md`。

验证完成：共同8条观测 MAE 从5.531降至1.739ms，MAPE 从19.454%降至6.152%。
但 high-skew 新模型选择的 LPT 比旧模型 reverse-odd 两轮慢1.578/1.576ms，
约5.69%/5.68%，无 trace E2E 慢4.26%；其余两条 route 计划相同。两模型都只
预测约0.07–0.09ms 的顺序差异，而实测约1.58ms。绝对精度改善不能替代排序
精度；保留负结果，用户实验基线不因此自动改变，不重新拟合。

后续只读 trace 诊断定位到两层：新模型将16T lane32置为关键路径，而 LPT
实测62/62次在1T lane69结束；自身顺序不变的1T队列也有明显 W13/W2 增时。
24–26ms 的 M2/M3 任务在较少总活跃核、更多小M竞争者下明显减速，模型的
16T队列进入 M1 尾部又比实测晚约4ms，导致预测竞争组成错位。此为描述性
证据，不区分 isolated/联合尺度的具体误差，也未识别 LLC/DDR 物理瓶颈。
详见 `optimizations/fused_moe_sve/results/order_differential_20260910.md`；
不修改压力函数或引入 oracle 到 planner。

2026-09-10补齐真实 expert M1–11 的独立1T网格，统一 CPU316、目标先执行、
其余任务全部等待、early merge关闭。两轮各31次，冻结模型 W13 MAE51.012us、
MAPE9.353%，W2 MAE32.175us、MAPE10.466%。M1/2低估约20–28%，M7/8的W13
在0.6%内，M9/10再次有约9–13%低估；不能用统一乘数视作已修复。880个
含预热/正确性调用的隔离目标均无后台重叠，实际CPU316验证通过；不拟合或
更换模型。完整协议和逐M结果见
`optimizations/fused_moe_sve/results/small_m_complete_isolated_20260910.md`。

### M1–11 的统一可控竞争响应（Lab，2026-09-10）

固定前台1T CPU316，分别测 W13/W2；背景均为同LLC的1T W13循环，比较无背景、
16/38个M2、16/38个M120，以及38核中19+19混合。相同核数下B分配空间相同，
每核4份8MiB权重；M120在一次调用的10个M12 panel中逻辑复用B，M2每次单panel。
该干预同时改变计算与A/C工作集，不能解释为纯B带宽变化或计算/访存固定比例。

每个前台M/stage使用本harness自己的无背景基线，报告同round的增量和比例；
不与真实expert独立点混算。两轮各5warm+31measured，132格smoke和9504次正式
调用全部完成。38核时所有11个M、两个stage的小背景减速都高于大背景，两轮
各自区间均同向排除零；M1/2 W13增时约210%对130–132%，M11约11%对3%。
16核下不少较大M响应很小，W2部分负增量完整保留。仅核数不足以表达该实测
类型差异，但本次不拟合/导出cost model，也未覆盖真实16T背景和阶段转换。
完整数据见 `optimizations/fused_moe_sve/results/small_m_controlled_pressure_20260910.md`。

### 有效计算/访存重叠原型（Lab，2026-09-10）

现有底层的 `max(C,D)` 已表示完全重叠。新增独立 Lab 形式：

$$
C=C_{ref}/\eta_c(M,s),\quad D_0=D_{ref}/\eta_d(M,s),\quad
T=C+g_s(P)D_0-\rho(M,s)\min(C,g_s(P)D_0).
$$

约束 $0<\eta_c,\eta_d\le1$、$0\le\rho\le1$，并以自身无背景时间 $T_0$
约束 $C+D_0-\rho\min(C,D_0)=T_0$。搜索范围 $C_{ref}\le C\le T_0$、
$D_{ref}\le D_0\le T_0$、$C+D_0\ge T_0$；不增加自由启动常数。
$C_{ref}$ 为包含前端/L1的既有M12代理，$D_{ref}$ 为层级端点时间代理，因此
效率与重叠参数只是有效模型分量，不能当作独立测得的硬件占比。

背景先用 $x=(n_{large}+w n_{small})/38$ 和 $g_s=1+a_sx+b_sx^2$，仅用
session1/M1的16/38纯背景拟合，M1归一化假设为访存主导。前台用奇数M与
纯背景拟合，同代理签名的偶数M不参与拟合；第二轮与所有混合条件留出。
这是既有数据的回溯验证，不是新的盲测。204留出格的部分重叠 MAPE2.035%，
重标定完全重叠对照2.533%，串行加权3.786%。混合44格仍全部低估，MAPE5.319%，
且不优于串行4.255%；以混合M1实测压力作oracle可降至1.791%，但不是可部署预测。
保留原型，不修改当前planner基线。具体参数/域/验证见
`optimizations/fused_moe_sve/results/small_m_overlap_20260910.md`。

### 六类 kernel 的访问历史函数（Lab，2026-09-10）

采用六个共享函数形式的实例 $F_g(r,s,h,p)$，$g=\lceil r/2\rceil$、
$r\in[1,12]$ 保留精确有效行数，$s$ 区分W13/W2，$h$ 是同一B的前序完整panel
数量，$p$ 为显式传入的访存服务倍率。冷/过渡/热是历史参考标签：0、(0,8)、
以及8以上，不表示已观测的缓存层驻留。它们不对应18个固定耗时。

无背景状态曲线用新测的1–12行、历史0/1/2/4/8/15构建；session1奇数行加完整12
在0/1/2/8/15上校准，保留历史4、偶数行2/4/6/8/10及第二轮。比较冷值常量、
0/1/8三个锚点插值和连续历史插值，后者允许非单调过渡，不强制B越热越快。
每个stage有独立参数；完整12单独保存，不能把M11的尾行store等同于M12。

任意正整数 $M=12q+r$ 可组合为 $\sum_{i=0}^{q-1}F_6(12,s,h_0+i,p)$，
若$r>0$再加$F_{\lceil r/2\rceil}(r,s,h_0+q,p)$。超过15个历史panel需显式允许
平台期外推；模型不会自动判断竞争导致的逐出，调用方必须更新/重置历史。

保留前一重叠原型的归一化C/D/重叠系数，以新历史基线共同缩放C和D；这是
有效历史修正，不把全部预热效应归于B。新测数据只有无背景，故$p>1$时的
热状态与压力交互明确标记未验证，不能直接作为生产planner的已验证预测。
218个留出条件上，冷值/三锚点/连续历史的块级MAPE分别为5.203%/2.704%/0.808%，
完整M的MAPE为1.276%/0.434%/0.344%。完整M的MAE为58.821/12.386/12.810us：
连续历史的总体绝对误差略高于三锚点，不声称所有指标均改善。两轮共10512次
含预热及smoke的调用均通过数值检查；6项局部测试通过。当前planner基线不变。
源代码为 `kernel_state_model.py`，报告为
`optimizations/fused_moe_sve/results/six_kernel_state_20260910.md`。

### 冻结无竞争基线的条件竞争增量（Lab，2026-09-10）

后续M1交叉测量否定无竞争历史倍率与外部竞争普遍独立相乘。按用户要求保留
原无竞争函数 $B_s(r,h)$ 及其参数，仅拟合
$T_s(1,h,n_S,n_L)=B_s(1,h)+\Delta_s(h,n_S,n_L)$，严格满足
$\Delta_s(h,0,0)=0$。非零压力暂仅支持M1、$0\le h\le8$、$n_S+n_L\le38$。

令 $n=n_S+n_L>0$，使用
$\Delta_s=(n_S/n)d_{S,s}(h,n)+(n_L/n)d_{L,s}(h,n)$。
纯背景增量在核数0/16/38及历史0/1/8间分段线性插值；零核数锚点固定为0，
其余24个锚点由第一轮纯背景实测减去冻结基线得到。它们是有效时间增量，
不能解读为已识别的C/D分解；无竞争误差不会通过改截距被消除。

第一轮混合与第二轮竞争共36点作回顾性留出，独立相乘/条件增量的MAE为
63.573/18.023us，MAPE为12.674%/2.991%。12个混合点仍全部低估，条件模型
MAPE7.283%；原无竞争对新实验的MAPE8.279%保持不变。训练24锚点精确插值
不是泛化证据，内部历史/核数插值及其余行类别未验证。当前planner基线不变。
实现为 `conditional_pressure_model.py`，详情见
`optimizations/fused_moe_sve/results/conditional_pressure_refit_20260910.md`。

### 固定19+19混合背景增量（Lab，2026-09-10）

保留上述无竞争与纯竞争曲线，针对已有19个M2加19个M120背景增加
$T_{mix,s}(h)=T_{linear,s}(h)+a_s+b_sh$，仅支持M1、历史0–8。
W13的$(a,b)=(45.073026,4.285658)$，W2为$(27.516140,2.671842)$，单位us。
每stage用第一轮历史0/1/8三个混合点拟合两个系数；第二轮6个混合点回顾性
留出MAE由47.344降至1.847us，MAPE由7.307%降至0.301%。全部第二轮30个
竞争点MAE为3.182us、MAPE0.736%。无竞争和纯竞争返回保持逐字段一致。
这不是其他混合比例的验证：非19+19混合请求拒绝，未测历史插值不声称已验证。
当前planner未接入。见 `optimizations/fused_moe_sve/results/mixed_pressure_refit_20260910.md`。

### 混合比例与中间历史的前瞻验证（2026-09-10）

冻结参数后新测38总竞争者、M2数量0/8/19/30/38以及历史0/1/2/4/8。
预先固定推广项$4f(1-f)(a_s+b_sh)$，两轮全部混合MAPE3.763%，优于无修正
线性混合8.290%；但历史2/4的24个混合点全部低估，MAPE6.689%。旧19+19
历史0/1/8复测MAPE0.605%，新比例在旧历史点MAPE2.416%。纯竞争历史2/4
本身MAPE6.721%；诊断性替换同期纯端点后，中间历史混合MAPE降至2.568%，
仍有W13/history4/8+30额外误差。故不能宣布比例推广与历史插值普遍成立，
本轮不重拟合、不改planner。详情见
`optimizations/fused_moe_sve/results/mixed_pressure_prospective_20260910.md`。

### 顺序修复纯竞争历史与混合增量（Lab，2026-09-10）

冻结无竞争模型及纯竞争历史0/1/8，在38竞争者下用第一轮纯背景补齐历史2/4
的8个修正锚点，之后固定纯背景，再拟合第一轮5个历史乘3个混合比例乘2stage
的30个混合残差节点。纯修正按比例加权，混合残差在纯端点严格为0，比例及历史
间分段线性插值。第二轮为回顾性同条件留出，不是新的前瞻验证。
纯h2/4 MAPE从6.734%降至0.662%；混合h2/4在原模型/只修纯/两层修复下
MAPE为6.824%/2.553%/0.434%。全部50竞争点最终MAE3.740us、MAPE0.630%。
对原修正30点改善、12不变、8变差；最差最终误差12.140us。无竞争预测保持
逐字段一致。38个新自由节点精确对应38个校准观测，不能以此声称未测比例/历史
泛化。38总竞争者外的纯背景沿用旧模型，其他总数混合拒绝，planner不变。
见 `optimizations/fused_moe_sve/results/history_mix_repair_20260910.md`。

### 修正版内部节点的前瞻验证（2026-09-10）

冻结修正版，新测历史3/6和M2/M120比例12/26、26/12（总38）。80个新条件中位数
MAE7.371us、MAPE1.253%，32低估48高估，整体未重现旧版6–7%的普遍低估。
但26/12的20点全部低估（MAE7.794us、MAPE1.242%），12/26的20点全部高估
（MAE8.078us、MAPE1.450%）；仍有局部比例偏差。最大新点误差20.882us，
来自W2/history6/12+26的高估。旧节点复测MAPE0.770%。不使用本轮数据回调参数，
不改planner；见 `optimizations/fused_moe_sve/results/repair_interpolation_20260910.md`。

### 修正版M1响应的planner局部接入（Lab，2026-09-10）

仅替换整expert M1/1T的GEMM阶段基线与压力倍率，使用冻结history0响应；未拆分
大M的panel历史，故history2/4修正本次不触发。竞争近似为同LLC活跃GEMM核数，
peer M<=12归small、其余归large，宽team按核计，W2 peer沿用W13背景响应，
39核封顶38。混合增量按n/38缩放。这些均为显式Lab外推，非测量支持的普遍规则。

三组各141形状、422DAG的mean-best/early-mergeoff完整搜索：高偏斜同形状LPT
但24个M1 expert换分配，实测两轮计算时间降低2.193%/2.418%；其余两组计划
完全相同。共同4计划×2轮MAE由1.655降至1.638ms，修正版仍把较慢基线计划
预测为较快，不能由选中计划提速声称排序模型已准确。模型改变候选任务分配，
同预算不意味着同DAG集合，联合候选重排反而会选回较慢计划。当前基线未切换。
见 `optimizations/fused_moe_sve/results/planner_repaired_pressure_20260910.md`。

### 新版Lab取消宽窄team残差倍率（2026-09-10）

按用户要求，后续新版Lab采用 `planner_no_team_residual.CorePressureModel`：
在新内存校准对象中清空wide_team_pressure与narrow_team_contention_correction，
所有宽度的独立/满并发倍率和窄team修正均为1。资源竞争、spill、明确开销和
M1局部响应保留；原校准文件/惩罚版保存对照，生产默认不变。

固定两份高偏斜计划后，预计31.139/31.285ms变为26.558/26.284ms，排序方向
与实测一致但总时长仍低估。16T平均偏差为-1.737/-1.816ms，W2计算接近实测，
剩余包括约0.98ms gather、0.34–0.36ms未分配间隙和约0.4ms W13低估。
1T平均仍低估约2.40ms：kernel阶段低估被约2.95–2.98ms旧operator残差部分抵消，
不能把残差直接视为真实开销。M1–4联合阶段误差最大；trace不能唯一分离独立
成本与竞争错误。本次不重搜、不重测硬件、不拟合补偿系数。
见 `optimizations/fused_moe_sve/results/no_team_residual_20260910.md`。

### 无team倍率基线上显式加入16T gather（Lab，2026-09-10）

`planner_gather16.CorePressureModel`在每个16T expert的W13前加入gather_pack_a，
$t_g=4.220+6.012680(\lceil M/16\rceil-1)$ us，支持H4096、M1–1945。
使用第一份计划第一轮63个expert gather中位数拟合两参数；其余195个观察回顾性
留出MAE5.924us/MAPE14.162%。这是联合条件下的阶段延迟，不再乘竞争倍率、
暂不额外注入共享访存流量，故尚不能代表gather对其他任务的带宽影响。

原GEMM阶段及1T/其他宽度不变，宽窄team倍率继续为1。固定计划的16T平均
低估1.737/1.816ms减至0.910/1.000ms；总体预测26.894/26.952ms仍低于实测，
相对排序再次错误。显式gather改善归账但没有修复小M联合响应。本次不重搜或
新测硬件；见 `optimizations/fused_moe_sve/results/gather16_model_20260910.md`。

### 实际8T条带的协同历史采集（2026-09-10）

新增真实8线程共同执行同一expert的采集：W13每线程128个GEMM N列/1MiB B，
W2每线程512列/512KiB B，full-stripe `(8,0,0,1,1)`。测尾行1/2/4/8/12、
前序完整块0/1/2/3/7及无背景/4个8T M2/4个8T M120背景；两轮10950条件
全部数值/几何验证通过。线程自身服务与team包络分开，前序panel间不加barrier。
W2无背景1行尾块h0约49–50us、h1约50us、h7约41–42us；h7完整12行约77us，
小尾成本远高于按2/12计算行比例缩放。W13八行h1比h0慢10.6%/13.3%，
表明历史响应依赖行形状/阶段/宽度，不能直接移植1T曲线。未拟合或改planner，
不由容量推断实际缓存驻留；见 `optimizations/fused_moe_sve/results/eight_team_history_20260910.md`。

### 8T 精确尾块基线与限定竞争增量（Lab，2026-09-10）

对 $M=12q+r$，$T_{8T,s}=\sum_{h=0}^{q-1}F_{12,s}(h)+[r>0]F_{r,s}(q)$。
第一轮无背景50节点冻结为 $F^0$，随后100背景节点拟合
$\Delta_{r,s,b}(h)=F^{bg}_{r,s,b}(h)-F^0_{r,s}(h)$，不再按M12计算行比例缩放尾块。
W13/W2 分离；r支持1/2/4/8/12，h节点0/1/2/3/7，区间内线性插值，不越界外推。
Lab adapter仅替换M≤96且精确尾行受支持的8T expert，其余保留旧模型；
分块保持计算/访存工作总量。未测尾行的覆盖边界存在不连续，不用于完整搜索。
竞争仅在同LLC恰有四个8T W13 M2或M120 peers时使用测量增量，否则仍用旧竞争项。

第二轮无背景块MAPE2.663%；背景块MAPE4.923%→2.840%，但整段MAE6.004→6.092us。
固定median/uniformish trace中W2 M13–24支持子集MAE36.69→15.39/35.30→9.09us，
8T lane MAE2.274→1.614/2.775→1.792ms；W13、M12及部分更大M回退。
两份trace的新竞争描述命中均为0，不能归功于新竞争项；微基准W2无真实W13继承状态。
不切换当前基线、不做新搜索或硬件采集。详见
`optimizations/fused_moe_sve/results/eight_block_model_20260910.md`。

### 单expert完整路径的8T无竞争桥接（Lab，2026-09-10）

目标expert移到真实计划首位，其余任务依赖其完成；实际gather→W13/SiLU/packC→W2，
8T CPU312–319，1MiB/512KiB条带、early merge关闭，两份输入24条件各两轮。
第一轮完整包络精确M查表预测第二轮MAE4.918us/MAPE1.035%；已有精确块分开模型
补相同第一轮非GEMM剩余量后MAE15.348us/MAPE2.418%。同M跨输入MAE5.688us。
已有精确块基线自身W13/W2 MAPE3.446%/1.842%，未复现联合trace的W13大幅高估。
完整路径标定更贴近该运行条件，不说明必须取消阶段分解，也未唯一识别竞争误差机制。
M12等总时间有回退/波动；未测M泛化和联合竞争仍未验证，不替换planner默认或实验基线。
详见 `optimizations/fused_moe_sve/results/eight_single_expert_20260910.md`。

### 8T分阶段DRAM需求表（Lab，2026-09-10）

独立stage测量定义 $p_s=(R^{read}_s,R^{write}_s)$，按M、W13/W2、实际owner条带及
权重复用状态条件化，另保留每调用流量与P10/P90。两轮DDRC read/write采集及无PMU对照：
8T轮换32份B的W13 M1/M12/M48约61/47/26GB/s；重复一份B全部条件低于0.5GB/s。
W2模式不同，M1存在较大轮内波动；连续/stride7路由差异未跨两轮稳定复现。
这些是独立的已服务DRAM需求代理，不是排队延迟或供给上限。未测M不插值，未将副本数
映射成panel历史h，也不把平均压力视为瞬时压力。联合使用仍需验证缓存状态与请求节奏反馈。
无planner/基线切换；见 `optimizations/fused_moe_sve/results/eight_stage_demand_20260910.md`。

### 固定8T/M12成本的需求加权竞争响应（Lab，2026-09-10）

固定真实完整路径第一轮成本 $T^0_{13}=161.765$us、$T^0_2=82.720$us，
冻结上一轮独立需求表。$x=\sum_j(p^{read}_j+p^{write}_j)/47.15856035147088$，
$T_s=T^0_s(1+a_sx+b_sx^2)$，非负系数、无拟合截距；无背景必回到原成本。
W13 $(a,b)=(0.018287133,0.000393287)$；W2 $(0.020100800,0)$。
第一轮同类32副本后台训练，混合/单份复用和第二轮全部留出；任务数对照参数预算相同。
第二轮28联合条件MAE2.936→1.083us、MAPE2.293%→0.882%；混合仅1.230→1.155us，
部分混合点回退。无背景约1%协议残差保留不拟合。压力和不是实测联合流量守恒等式。
仅验证同LLC、前台8T/M12、持续后台；其他M、动态阶段切换和真实trace尚未验证，
不切换planner基线。见 `optimizations/fused_moe_sve/results/eight_joint_response_20260910.md`。

### 冻结需求响应的跨M与动态事件验证（Lab，2026-09-10）

冻结全部成本/系数与需求，测前保存自主事件预测；有限8expert批次，W13→W2真实packed C依赖。
M8仅固定后台前台预测，无缺失需求插值；M12/13/48自主推进所有team的阶段与结束事件。
新M静态需求模型前台完成MAPE5.06%，任务数5.27%；M8单独仍11.88%。
动态需求模型前台/总体完成MAPE1.13%/1.04%，任务数0.94%/0.53%，未全面胜出。
对同协议无背景作诊断差分：M12/M13/M48实际批次增量约102.90/185.99/143.40us，
需求预测112.84/137.60/225.87us。直接误差包含基线与竞争误差抵消，不能视为响应推广成功。
2976个动态前台阶段中2519个跨后台阶段边界；验证了混合重叠但未重新拟合或切换基线。
后续需前台M/块敏感度及阶段状态/间隙验证；见
`optimizations/fused_moe_sve/results/eight_response_transfer_20260910.md`。

### 前台块／尾块的非线性需求响应（Lab，2026-09-11）

保留全部阶段成本和原需求条目，独立补M8需求。以旧F块曲线分配成本并归一化，
$t^0_{s,b}=T^0_{s,M}F_{s,b}/\sum_bF_{s,b}$，确保无背景总成本不变。
首个12行块响应保持原系数；W13/W2各新增reuse12(h1..3)、row8(h0)、tail1(h1)三条
$R_{s,g}(x)=1+a_{s,g}x+b_{s,g}x^2$，非负系数、无自由截距。
块级相对变化仅用完全处于1/4个W13-M12后台覆盖内的训练样本，不用绝对残差回调成本。
曲线冻结后两轮新鲜无额外panel计时留出：静态/动态竞争增量MAE51.28/77.17→21.43/18.27us。
但动态全部team绝对完成MAE85.47→183.08us，基线、间隙与静态M48残余仍未解决。
保留曲线响应候选，不切换整体planner；未知尾行/历史不外推。详见
`optimizations/fused_moe_sve/results/eight_block_sensitivity_20260911.md`。

### 联合资源模型的基线闭合（Lab，2026-09-11，进行中）

`analyze_joint_baseline.py`在真实单expert trace上定义可加核算：
`T_expert = T_gather + T_W13 + T_W2 + G_before_gather + G_before_W13 + G_before_W2 + G_after_W2`。
逐样本核对时间端点、阶段不重叠和总和，均值保持可加；独立中位数不具有该性质。
已有24条件两会话GEMM重复MAPE为W13 0.3914%、W2 0.4722%，总时间1.0353%，
不能据不同状态的连续GEMM链偏差直接回写真实路径T0。
新增两会话完成2084次调用校验；历史同形状session1预测新两会话，W13/W2 MAPE为
0.5141%/0.4200%，总时间1.1632%。这支持真实路径重复性，不证明未见M泛化。
新模型计划先闭合协议匹配基线，再引入状态需求、共享服务容量及重叠反馈，最后验证混合宽度
和未见联合计划；目前为基础组合与worker诊断候选，没有新增已验证资源公式或更换planner默认。
Lab基础组合候选`joint_baseline_model.py`定义`P_s(b)=T_s(12b)`、
`D_s(r)=T_s(12+r)-T_s(12)`，主假设`T_s(12b+r)=P_s(b)+D_s(r)`，r=0只取P；
M<=12直接保留首块逐行成本。完整前缀已含stage setup，不能重复添加。
历史缩放尾块仅作诊断对照，两个版本均只拟合训练session1，域限8T/M1..60真实路径。
gather非负仿射与间隙常数分别标定；这些尚未通过留出，不代表可靠资源模型已完成。
训练两会话2406次调用校验完成，主版本session2完整expert MAPE1.4642%，P90误差3.8601%，
14个后续尾块留出预测已冻结并完成验证，结果如下。
原始worker trace进一步确认gather包络混入到达错位：M1包络45.79us、最大worker区间1.87us，
M12分别56.53/5.395us（两会话中位数均值）。单样本应按
`E_gather=max_i(a_i+g_i)-min_i(a_i)`核算；不能将整段包络作为资源服务时间或对外需求。
现有gather仿射只限isolated-first-expert对照，初始worker到达与后续任务状态仍需建模。
14个后续尾块留出已完成1284次调用校验：主版本完整expert MAPE1.8896%、P90误差3.9385%，
但W2 MAPE5.8632%且全部条件高估，平均+13.9243us；不能由总时间通过推断阶段模型通过。
完整块比例缩放对照W2仍4.5071%，保持候选不采用，需辨识前缀与尾块的历史残差。
新增worker时间可选解析及同核首任务/先行M36后任务对照；先行任务严格串行，其他任务
仍必须晚于target结束。实际到达标记只作诊断，不作为模型预测输入。
同核首任务/先行M36后任务两轮1042次调用已验证：目标完成减少约38.78–47.05us，
gather最大worker区间仅变化-1.68至+1.05us，到达跨度减少约35.97–42.44us。
W2变化-1.615至+0.740us，不能解释尾块留出13.9243us偏差。启动/到达项与
GEMM历史残差需独立处理，隔离首任务的到达跨度不能逐expert重复收取。
同expert/同packed-B/嵌套route前缀对照1968次调用已验证：W2的1行整阶段净增量在
h1/2/3/4为49.085/38.210/29.515/21.515us，11行h1/2/3为83.455/80.330/79.525us。
因此历史响应依赖kernel类别；仍是跨M整阶段差分，未声称真实panel内部计时。
其余3/5/9行及首块的同权重补采进行中，未据此切换默认模型。
初始时序候选定义`start_i=max(R_task,A_i)`、`finish=max_i(start_i+G_i)`，避免逐任务
重复添加启动时间。固定CPU中位入口表在跨计划诊断中单次MAE约23us，最早入口中位数
偏差约31us，不能视为已验证的完整到达模型；当前只实现校准/时序不变量，尚未接入planner。
六类kernel历史候选已进一步采用`D_g(h)=a+b*exp(-lambda*(h-1))`（非负参数、h1..4），
用奇数行代表拟合，偶数行仅加首块配对差；a不解释为物理计算下限。
18点两会话前瞻验证1640次调用通过，W13/W2/GEMM MAPE分别0.6593%/0.5252%/0.4581%，
P90相对误差1.3631%/1.1121%/0.9822%。这是8T/M<=60/固定窗口的基础GEMM证据，
不等于联合或其他宽度已验证。此前真实route回顾性迁移W13/W2 MAPE为0.3904%/0.4741%。
Gather整数工作映射已与native的300配置对齐，算法读写量不能冒充缓存/DRAM流量；8T
worker区间线性候选形状检查MAE1.8133us、MAPE15.9391%，仍需与到达/共享服务分开验证。
Lab新增进程局部JIT getter包装以观察真实panel区间，原kernel不变，无panel barrier。
按实际调用序列和B指针验证数据，预设同进程control/observe阶段中位扰动<=1%；
两场原观测版1066次调用/21760条panel通过完整性校验，但W2/M49阶段配对扰动
为+1.949%/+2.079%，多个M重复超过1%，拒绝用于成本或资源校准。避免重复发布相同
kernel指针的变体另两场也未通过：W2/M25为+1.686%/+1.231%，M49为+0.653%/+1.519%。
计时/记录/门槛不变，后续独立定位时间戳和记录开销。逐worker配对分解显示缓存发布版
24个M/stage/session组的last-finisher到达偏移均值变化绝对值<0.1us，不能将主要差异
归为team入口错位。Lab count-only编译消融保留包装/CPU查询/计数，关闭时间戳和事件payload；
独立日志类型禁止其作为panel时间数据。该消融两场1066调用/21760拦截校验通过，
24个GEMM组中22组满足1%，但第二场W2/M12为+1.065%、M49为+1.398%，仍不通过整体门槛。
去掉时间戳和payload并不足以消除全部扰动；下一候选为稀疏worker/panel采样。
单worker候选按call ID轮换CPU312–319，未选worker直接调用原kernel；仍按实际8T N切分验证，
另检查同pair同CPU的worker区间扰动<=1%，防止team最大值掩盖单worker减速。分次采样不得拼成
同时team包络，完整native阶段trace负责全team阶段先后检查。两场稀疏实验1066调用/2720panel
校验通过，但team仅22/24组、被采样worker仅15/24组满足1%；M13/W2 worker为+2.155%/+1.779%。
拒绝将其校准为真实panel成本。观测变体保留诊断，独立共享服务实验继续使用已验证whole-GEMM
无竞争基线，不能把历史净增量参数解释为物理计算下限。尚未加入PMU或将
panel观测作为已通过的资源模型证据。
验收预设为未见计划MAPE<=3%、P90误差<=5%，固定实测候选集选择损失median<=2%、P90<=5%，
首批真实locality对照固定前台expert13/M13/8T及后台expert37/233/151/157（M48/24/12/12），
启用0/1/2/4后台，同LLC CPU280–311与跨LLC CPU248–279对照；前台CPU312–319。
后台按真实gather→W13→W2推进，其他任务等待观测cohort全部完成。零背景也保留相同任务配置，
仅让后台等待前台完成；不把后台数量或local/cross差直接当独立LLC/DRAM需求或容量。
首版提升head遗漏旧lane前驱旁路，烟测发现NaN；补齐依赖并验证所有共享核任务对有DAG先后，
修复版两场738调用校验通过。零背景冻结T0相对偏差W13为+0.745..+1.023%、W2为+0.104..+0.737%。
四后台时W13配对增量44.05..47.56us，W2为1.44..2.79us；local与cross均出现主要W13减速。
前台W2期间实际仅约两个后台活跃，不能将整段四后台常数用于W2，也不能独立归因前台敏感度。
这仍是M13动态cohort诊断，不是资源容量识别或未见M泛化验收；模型参数尚未改。
同时要求分组偏差、噪声和相同搜索预算检查。报告：`results/joint_cost_model_20260911.md`（MoE Lab）。

**首块请求窗口候选（Lab，条件时间线）。** 固定8T/M<=60的上述无竞争模型，令
`tau_s(M)=T0_s(min(M,12))`，一个stage在该参考窗口内请求Q13=8MiB或Q2=4MiB，
剩余`T0_s(M)-tau_s(M)`作为不竞争本资源的服务时间。这里Q是packed-B需求代理，
不是实测DRAM流量；首窗口也不是已验证的真实panel时间分解。
活跃请求窗口的参考速率`r_i=Q_i/tau_i`，容量C下先取
`u_i=1/(1+alpha*sum_peer(r)/C)`，再取`v_i=u_i*min(1,C/sum(r*u))`；
窗口按v_i推进，剩余服务按1推进，故减速会延长请求窗口并反馈到后续重叠。
alpha=0为纯容量对照。无竞争严格恢复T0；若C低于单个任务参考速率则拒绝参数。
W13结束后经过固定gap再启动W2，两stage均自主推进，不读取实测W2起点/终点。
首版仍以每个活跃team的实测W13起点为条件，因此不含gather/入口预测，不能作为完整planner模型。
只拟合M13 session1：纯容量C=202.5GB/s等效单位；低压力响应版C=200、alpha=0.08；
gap=0.39us来自零背景。训练16个阶段条件中位数MAE分别2.1843/1.3914us。
这两个容量是需求窗口模型的有效参数，不是硬件带宽测量。M12/M48采集前已冻结绝对/相对增量
迁移对照；动态候选在新数据开始采集后、读取其时间前冻结，只称拟合留出，不称测前冻结验证。
M12/M48四场1476调用校验通过。有竞争条件下低压力版目标双GEMM跨度MAPE0.925%/0.639%，
cohort跨度MAPE0.337%/0.570%，均仍以实测初始W13起点为条件；不是完整计划预测精度。
M12/W13最大阶段误差3.513%，M48/W2较纯容量版MAE0.683→1.745us回退，不能只报告有利项。
保留候选和对照，下一步移除实测起点并验证gather/ready及全动态时间线，不替换planner基线。

**入口/gather组合候选。** 对初始无依赖且核不重叠的8T cohort，逐worker gather服务
`G_i(M)`来自既有逻辑工作量模型。每个旧入口场景保留同一次调用80CPU的向量A，
以`W13_start=max_i(A_i+G_i(M))+g_setup`启动请求窗口模拟，最终取各场景输出的中位数。
对照先取每CPU入口中位数，再计算team最大值。g_setup=2.28us仅从M13第一场零背景阶段间隙获得；
resource C/alpha/gap、GEMM基线、gather系数及旧入口样本均不重拟合。现在不输入目标运行的实测时刻，
但gather尚无共享资源反馈，仅支持初始cohort，显式拒绝依赖任务和非8T/重叠核集合。
回顾性correlated版本M13/M12/M48目标完成MAPE0.705%/1.805%/2.085%，cohort完成
0.789%/0.818%/2.203%；起点为scheduled_compute，非最终merge或全计划完成。
M48第二场入口共同延后使起点误差显著，不能新增M48残差掩盖；旧入口场景分位数不自动等于校准置信区间。
冻结M17/M35完整数值预测后再采两场新条件，correlated预先指定主版本、per-CPU median为对照。
四场1476调用校验通过，主版本M17/M35前台完成MAPE1.485%/1.073%、P90为3.159%/1.923%；
cohort完成MAPE1.134%/0.935%、P90为2.130%/1.640%，通过初始cohort完成时间门槛。
W13起点MAE8.570/6.542us仍较大；入口场景带对cohort样本覆盖率仅58.5%/71.2%，不能用于
声称已校准置信区间或安全剪枝。此结果不覆盖依赖后继、大M>60、其他宽度或完整planner搜索。

**依赖驱动队列候选。** 请求窗口模拟新增按DAG释放任务：`R_j=max_{p in deps(j)} end_W2(p)`，
root取0；释放后逐worker取`start_gather_i=max(A_i,R_j)`，gather结束和既定setup决定W13起点。
因此初始入口不在每个后继上重复计费，W2/后继起点均由预测完成事件触发。共享请求速度、
T0、gather系数和所有resource参数保持冻结，gather仍未竞争本资源。依赖必须按输入顺序拓扑有序，
任意共享worker的任务对必须通过DAG祖先关系排序；无依赖结果与前一cohort预测逐字段相同。
首批15个真实expert分五条8T队列，每条3任务，M8..60；比较forward/reverse/rotate/staggered，
以及joins/joins_reverse/serial控制。其他原任务等15任务全部完成才执行，故仍是全计划的受控前缀。
测前冻结完成预测约1.996..2.033ms（四队列顺序）、2.654/2.659ms（joins）、8.566ms（serial）；
不读实测时刻、不重拟合参数，随后两场验证并检查实际依赖边与阶段先后。
首批稠密实验图的连续队列MAPE4.273%、P90=5.196%未通过；joins/serial为1.902%/0.947%。
trace显示同lane后继在gather前等待约40..50us，GEMM自身误差不是主要来源。原native代码在W2后
先barrier、发布task状态并逐个通知successors，再由本team越过最后barrier；模型把数据完成与
worker可复用合并成一个时刻，遗漏了该区别。实验给每个pilot接约209个剩余任务造成高fanout，
原anchor仅224任务/215边，因此不能把此间隙拟合为所有8T后继的常量惩罚。
配对传递约简保持每个节点祖先集合、pilot依赖、核/输入完全一致，边数3345→255等。
两场1066调用通过数值/trace校验，同lane关键交接间隙中位43.47→3.01us；连续队列实际缩短
67.42..92.77us，冻结模型MAPE4.413%→0.787%，P90=1.537%。joins/serial约简后为1.242%/1.188%，
没有同等收益，说明部分通知成本可被其他lane的工作隐藏。没有修改生产图或重新拟合参数。
后续需区分边通知就绪与worker退休时刻，并保留原稠密图反例；混合首波M8/W13仍有约10..15us
低估，约简并未消除该独立竞争响应问题。当前仍非完整planner模型。

**发布/worker退休候选。** 令task p的计算终点为E_p，完整native successor列表长度为d_p，
其中child j的零基rank为k_pj。候选`notify(p,j)=E_p+b+nu*k_pj`，
`reusable(p)=E_p+b+nu*d_p`。新任务gather逐worker从
`max(initial_entry_i, max_parent notify(p,j), previous_worker_reusable_i)`开始，保持通知与占用分离。
模拟器在W2结束时回调，记录未来通知/可复用时刻；completion仍取W2最大终点，不冒充最终merge完成。
publication启用时必须提供完整有序successors（包括未在受控前缀中执行的任务），拒绝缺失通知元数据。
未启用时保持原模型输出。C、alpha、T0、gather参数均冻结。
只用queue_reduction第一场三条链的1/210 fanout交接中位数2.955/44.115us，得到
b=2.758062us、nu=0.196938us/通知；这是80-worker/8T环境的有效模型，不是独立atomic指令延迟。
保持完全相同可达关系，构造fanout9/33/65的forward/reverse/joins/serial新图，测前冻结新旧预测。
模型预测串行在这些fanout下不随d增长（worker退休已被其他lane工作隐藏），连续链随d增长，
joins可能因错开竞争而非单调变慢；等待新两场验证，不把拟合端点当泛化证据。
新两场1066调用校验通过，发布版forward/reverse/joins/serial完成MAPE分别2.209%/1.653%/
0.717%/0.920%，P90均<5%；旧版forward为3.119%。Joins较旧版0.496%回退，不能只报告有利项。
局部gap仍系统低估：fanout9/33/65预测4.531/9.257/15.559us，实测中位4.955/12.130/23.460us。
因此保留通知与worker复用分离结构，但不宣称线性通知成本已可靠泛化，也不由完成门槛通过
推断M8竞争、独立资源需求、其他宽度/大M或完整planner已验证。

**等宽team服务分配对照（Lab，未采用）。** 在既有参考请求r与低压力速度u上，
令期望服务d_i=r_i*u_i。默认仍按比例分配；可选max_min在总需求超过C时选择lambda，
使`b_i=min(d_i,lambda)`且`sum(b_i)=C`，请求进度`v_i=b_i/r_i`。
总需求未达容量时直接b_i=d_i；私有服务速度保持1。仅比较相同8T宽度，不能据此定义混合宽度公平权重。
固定C=200、alpha=0.08、gap=0.39及T0的旧混合首波条件诊断：M8误差-15.387→+11.463us，
M12为+33.446→+8.364us；M16筛选子集为+11.433→-13.226us。公平分配不能单独闭合误差，
请求窗口/重叠假设仍待辨识；该诊断输入实测W13入口，不是自主泛化。
新前台M4/M8各8个0/1/2/4背景与local/cross条件，背景M48/24/12/12，使用旧完整入口场景
分别冻结两种分配的自主预测。产物`tmp/joint_cost_model_20260911/allocation_small_m`，
预测先于采集；当前未评分，不更改planner默认。独立需求、混合宽度和全计划验证仍欠缺。

M4/M8四会话1476调用校验通过，测前评分两分配的有竞争前台完成均未达门槛：
比例MAPE5.108%/8.469%，公平3.670%/6.515%；cohort均<3%，不能掩盖前台失败。
无竞争W13 MAPE0.465%/0.509%，W2 1.209%/0.771%，但M8入口到完成低估约19.683us。
1/2背景两分配预测相同且均低估；条件实测入口后M4 W13仍低估4.497/12.938us，
M8为3.132/9.860us。因此低压力响应也需辨识，不能仅改变容量饱和分配；本批若用于
后续拟合即转为开发集，另采留出。详见上述报告与allocation_small_m/evaluation.json。

**请求密度敏感度候选（Lab）。** 原比例分配中仅替换W13窗口alpha为
`alpha_i=0.08+beta*max(0,T0_W13(12)/T0_W13(min(M_i,12))-1)`。
beta=0保持旧模型，M>=12保持原alpha；无竞争peer需求为0，故不改变T0。
比值来自首窗口packed-B代理速率，不是独立硬件流量。beta只用M4 session1的n1/n2
条件W13拟合为0.632；W2、capacity、gather、到达、后台M>=12全部冻结。
新M6/M10数值预测先于采集保存于`tmp/joint_cost_model_20260911/density_small_m`，
必须按前台阶段及完成误差评分，不能让cohort最长后台掩盖小M失败。尚未完成留出评分。
另有回顾性独立anchor早期10调用预测后续21调用的入口校准诊断，M8零背景完成MAE
19.166→1.964us；不修改原前瞻分数，也不默认使用目标运行时刻。两问题分开验证。

请求密度候选M6/M10四会话1476调用通过数值/trace检查，但冻结验证未支持通用采用。
有竞争W13 MAE：M6 14.295→1.135us，M10 1.876→4.431us；M6 W2 MAE2.951→4.250us回退。
新前台完成MAPE8.942%/3.923%均未过门槛，入口误差独立保留。
实测入口条件诊断下M6 n4仍低估5.828us，M10 n1/n2/n4均高估约2.8..3.6us，
说明自主阶段分数可能有误差抵消，不能只选有利项。保留beta冻结反例，不加M阈值或事后重拟合。
下一步辨识计算/访存重叠与kernel响应转折；独立需求、其他宽度、大M、完整计划及planner仍未验收。

**独立首块时钟的重叠对照（Lab，未采用）。** 计算时钟c=T0_s(min(M,12))，
W13请求窗口q=min(c,tau)、W2为q=c；首块计算与请求并行，首块结束=max(计算结束,请求结束)，
随后执行T0_s(M)-c的原私有增量。c/tau只是有效时钟，非实测物理计算下限/服务分解。
资源速率为Q/q，请求完成后退出竞争；无竞争保持T0，tau=inf且alpha保持旧值时退化为旧模型。
以M4/M13第一场训练所有活跃expert W13，原网格最优C200/tau116/alpha13=0.2，扩展后
C260/tau140/alpha13=0.5。扩展训练MAE4.292→3.349us，但M10 W13迁移MAE
原2.790→10.524us，M6 W2原3.087→7.015us；不采用，不修改默认。
此为已观察数据的条件诊断，不能据此排除所有重叠结构。保留完整网格、残差和不变量检查于
`tmp/joint_cost_model_20260911/overlap_diagnostic`，下一步必须增强参数独立可辨识性。

**实际K循环类别条件响应（Lab，待验证）。** 原生BF16 exact-M生成器在rows<=8使用
physical_rows=8及双缓冲K循环，rows9..12切换physical_rows=12及分组K循环；并非任意M阈值。
首请求窗口r=min(M,12)仅当r<=8时令
`alpha_s=0.08+beta_s*max(0,T0_s(12)/T0_s(r)-1)`，否则保持0.08。
W13 beta0.632冻结，W2 beta0.458仅用M4第一场n1/n2拟合；T0/C/gap不变。
六个row-pair类别不意味着奇偶行完整kernel等同，也不能以静态分支证明物理瓶颈。
M6开发条件W2残差缩小，M10逐字段恢复原模型；M7/M9新鲜预测已冻结并启动验证于
`tmp/joint_cost_model_20260911/family_small_m`，尚未评分。后续尾块暴露、宽度及全计划仍欠缺。

kernel类别候选M7/M9四会话1476调用校验通过。冻结有竞争M7 W13/W2 MAE
11.523/2.369→1.364/0.929us；M9保持原模型3.018/0.799us，不引入额外修正。
但新前台完成MAPE3.891%/5.215%均未过门槛，零背景完成也因入口等误差未通过。
独立Lab DAG adapter传递原prepare/on_complete，24个一致性检查通过。旧15-expert
reduced队列自主回放完成MAPE0.966%→0.883%，M8首波W13改善、forward/joins后续
W13却回退（MAE5.648→15.842、4.762→22.465us）。保留kernel类别特征与反例，
不采用全局候选；需分开验证动态重叠与后续环境，不以整体平均分数掩盖阶段失配。

后续M8定位新增同波上下文反例：joins最终波与joins_reverse首波的expert/M/核完全一致
（M8/8/12/12/16），只模拟该波、输入实测起点且排除同时运行的其他任务后，family仍对
后续M8高估约20..23us。配对实测后续M8比首波快18..20us，其他三个任务也快约11..25us，
均在两场方向一致；模型只预测M8约0.3..1.4us变化。该差异不是前驱预测clamp造成的。
因此仅有静态kernel类别和当前活跃请求集合不足以闭合当前数据；需辨识可迁移的前序执行
状态对需求/服务的作用，不能按波次减常量或直接认定B历史因果。证据matched_wave.json。

固定目标波次上下文辨识已启动：M8/8/12/12/16始终在CPU280–319，以真实任务构造
0/1/2前序波；相同前序工作分别在同LLC或CPU240–279跨LLC执行，全部先于目标完成。
原任务集合与目标窗口保持，完整资源依赖静态验证；seeds611601/611602，数据与协议
`tmp/joint_cost_model_20260911/wave_context`。此为状态辨识实验，不新拟合模型；
位置变化仍包含局部worker/缓存与共享域作用，不能单独解释成DRAM效应，结果待采集。

前序位置实验两会话492调用通过完整校验。目标M8的W13在同LLC前序1/2波后
配对缩短约17.5..22.9us，跨LLC相同前序仅缩短约4.3..7.0us；直接cross-minus-same
差约10.4..17.3us，pair MAD约2.3..3.1us。cross起点并不更早，简单等待时长不能闭合该模式。
这支持局部执行上下文贡献，尚不能区分目标核与同LLC共享状态。下一步用目标核/同域其他核/
跨域的相同前序工作区分，不按波次加常数。证据wave_context/analysis.json，未改模型参数。

目标核/同域其他核辨识已启动于core_context：同五个前序expert统一两条8T依赖链，
只改变CPU280–295（M8目标核）、296–311（同LLC背景核）、248–263（跨LLC）的放置。
三种bridge仅前五个core_begin不同，目标五expert及完整依赖保持；无前序另作基线。
seeds611621/611622，结果待采集，不新增状态系数。peer核活动也可能改变背景需求，不能
仅凭目标速度差宣称纯缓存因果；各目标/背景阶段一起分析。

两lane前序位置对照两会话410调用校验通过：M8在目标核前序或同LLC其他核前序后
均缩短约9..11us，直接配对两位置差-0.190/0.000us，描述bootstrap区间均含0，未宣称严格等价。
两种同域位置相对跨域均快约4..6us，且从未执行前序的M16核也获得同域收益。
目标核自身活动不是此收益的必要条件，下一步以共享域执行状态辨识为主；尚不能单独归因缓存或时钟，
也不能把五lane与两lane的差简单当线程数系数。数据core_context/analysis.json与intervals.json，
无新状态参数采用，完整目标范围保持。

共享域历史代理诊断显示累计工作量不足：相同5expert/60MiB逻辑权重/1.598GFLOP的
40核与16核前序，局部收益约15..17us与4..6us；加倍前序工作没有加倍收益。
单条件比例标定的已观察条件迁移中，峰值分配核数优于累计工作和所测EWMA，但仅两种
并发水平，不采用参数或宣称物理机理。GEMM核区间也未包括native等待worker的状态轮询/yield，
不能视为实际全域请求压力。证据domain_history_diagnostic/results.json，下一步需新水平和独立服务验证。

共享域峰值代理的24/32核新水平已冻结验证：同五个前序任务分3/4条8T链，同/跨LLC配对，
目标波次不变。旧单条件系数0.427875us/core预测局部收益10.269/13.692us，累计任务数对照
均17.115us；测前文件mid_context/frozen_predictions.json，seeds611641/611642。
这是上下文收益代理验证，不是硬件带宽或全计划预测，结果待采集；未采用峰值状态公式。

24/32核前瞻两会话492调用通过，实际峰值逐调用核对。冻结峰值规则预测局部收益
10.269/13.692us，新实测分别7.550/8.865及10.585/9.985us；MAE2.734us，优于累计任务
对照7.869us，但四点均高估。保留共享域活动强度特征，不把它直接转成每任务时间减项；
状态衰减与资源响应的连接、其他形状/宽度和全计划仍未验证。证据mid_context/evaluation.json。

**状态到资源响应的条件对照（Lab）。** 固定T0/kernel响应，C_eff=C0*(1+gamma*P/40)，
gamma只由旧one_same/session1 M8上下文差拟合为0.088；或保持C0而只将W13 Q归一化为
Q/(1+gamma*P/40)。全stage同倍Q归一化与C增大代数等价，不凭时间数据声明物理容量/流量。
旧同域条件中W13特定压力候选使M8/M12/M16 W13 MAE19.398/27.695/16.521→
7.543/16.697/5.553us，M12/W2保持约1.43us；全stage C变化会使其W2回退至3.25us。
仍有系统残差，P相同的不同历史也未闭合；未实现动态状态衰减或跨域并发分配，未采用。
证据state_service_diagnostic，60隔离/100代数等价/5非法参数检查通过。

**自主状态DAG原型（Lab）。** 每域记录预测active expert核数及峰值；W2完成事件锁存峰值，
之后真正激活的W13按Q/(1+0.088*latched_peak/40)归一化。初始状态0，每预测call重置，
当前假设call内不衰减；不是已经辨识的物理状态。只支持既定NUMA3单LLC8T team。
activation必须发生在预测事件时刻，不能在未来任务准备时提前读历史；原依赖/publication继续保留。
30个零效应DAG及延后激活、域隔离、重置检查通过。旧15expert队列自主回放M8 W13 MAE
8.976→4.262us，forward/joins后续误差明显减小，首波保持；完成MAPE0.883%→0.921%，
总体基本持平。未见队列/衰减/其他宽度/全计划/选择效果未验收，不采用默认。
证据dynamic_state_candidate/results.json与identity.json。

自主状态的新顺序验证已冻结：同15expert/5x8T，10个未见队列排列加forward/reverse控制，
三模型在固定12候选中均直接选择new_06。seeds611681/611682，预测先于采集保存于
state_order_holdout；新顺序完成误差与集合内选择损失分别报告，不按同赢家重新挑候选。
范围仍是受控15expert前缀，非完整224expert或相同预算完整planner搜索，结果待采集。

新顺序前瞻两场1066调用通过：10个新排列的dynamic完成MAPE0.479%、P90=0.777%，
publication/family也约0.468%，不能宣称整体精度领先。首波M8阶段改善保持，后续M8
W13 MAE6.233→4.368us。三模型同选new_06，两场集合内选择损失0.260%/0.081%，差异小于
配对波动；不是广泛regret或完整planner搜索验收。支持保留候选，仍不默认切换。
实际原anchor224任务/12288路由中，当前支持域8T/M<=60仅2124路由；15pilot仅402路由。
下一优先级大M8T（7709路由）与16T（2455路由），路由份额不等于实测时间份额。
证据state_order_holdout/evaluation.json与coverage_audit/results.json。

大M8T基线扩展已准备并仅启动训练：真实expert85原M1205，同B/嵌套输入前缀、CPU312–319、
full stripes，12训练点与12留出点分开。主候选在M60连续，完整块每stage稳态斜率仅拟合
session1完整12行倍数；尾块沿用原h4独立曲线平台，M<=60不改。尾部平台与稳态斜率均待验证，
第二场作重复性，留出必须在模型/预测冻结后采集。产物large8_baseline，非联合/16T完成声明。

大M8T训练两会话1066调用通过，完整块稳态斜率W13/W2=150.469/72.293us每12行，
重复场整阶段MAPE0.239%/0.554%，旧M12/M60锚点未改变。但r5/h100整阶段配对增量
W13约69..71us而h4平台预测106us，W2约15..19us而预测37us；尾项平台不足，不能据整阶段
低百分比误差宣称分解准确。保留原候选反例，留出尚未采集，先补长历史各尾类训练再冻结修正版。
证据large8_baseline/model.json和tail_diagnostic.json，非独立panel测量、非联合验证。

长历史尾项补充训练已启动：同expert85/8T，在M600与M1176完整基点之后分别追加
r1/3/5/7/9/11，h50/h98各两会话。与原留出不重合，逐pair保留整阶段净增量和MAD，
不能把净差当独立panel服务或截断噪声。产物large8_tails，留出仍未采集，结果待完成。

长历史六类尾项四会话1312调用通过。W13 r1/r3在h50→h98仍下降；W2 r1整阶段
净增量四组约-10..-18us，描述bootstrap区间均不含0，不能当作非负独立尾kernel成本或截断。
完整M12块数未变，但阶段继承/布局/worker时序尚未独立辨识。下一候选以有符号形状净修正
表达全阶段时间，保留正总成本及M<=60原基线，禁止把该修正当作负资源需求或物理计算下限。
原留出未采集，先冻结修正版；证据large8_tails/analysis.json、intervals.json、worker_check.json。

大M8T修正版测前冻结：完整前缀斜率不变，尾部改为h4/h50/h98节点的有符号全阶段
形状净修正，h50/h98只用session1，history线性插值，最多延伸到h100/M1205。保留M<=60
原模型；全部1205形状通过正阶段时间/首块下界检查，不提供M单调剪枝保证。
原12留出加6个r8/r10点共18形状，预测冻结于large8_baseline/model_v2_frozen.json，
四会话留出串行采集中。不得将有符号净修正解释为负资源需求，完整联合和16T仍未验收。

大M8T修正版18形状留出四会话1640调用通过：W13/W2 MAPE0.183%/0.394%，双GEMM
合计0.153%、MAE11.846us，均达到本批门槛；原h4合计MAE26.624us。保留无竞争阶段候选，
不据此认定联合需求、gather/入口或完整计划通过。16T首批独立标定随后启动：expert96/M<=1718、
CPU304–319、owner512/256KiB，first_rows与bulk两批四场；尚未覆盖所有16T尾历史或拟合模型。
证据large8_baseline/validation_results.json和width16_baseline/protocol.json。

16T首批四会话2132调用通过，W13/W2重复MAPE约1.03%/1.03%；完整块每12行
斜率75.246/36.432us，第二场完整块MAPE0.384%/0.709%。仅为基线发现，不是任意M模型。
M13-M12净增量W13约69us，长历史M1718-M1716仅约8..11us，W2长历史差噪声较大，
不从两个点推物理负尾成本。h1/h2/h4/h140六类odd尾项训练已顺序启动于width16_tails，
不借8T参数，留出尚未准备或采集。默认planner保持原基线。

已冻结steady_background联合诊断：M8在M48同team前驱之后启动，0/1/2/4个大M8T
后台为1205/768/714/529。接入已验证大M基线后，首块请求+private模型在全部旧入口场景
预测后台请求先结束、W13仍持续，目标两GEMM均不减速（125.440/59.750us）。
实际阶段覆盖仍待测，不能把模型private区间当实测缓存状态。实验当前仅准备，等待16T采集结束；
用于检验持续请求缺口，不是全模型完成或DRAM机理结论。

### 大M gather多panel服务候选（Lab，尚未前瞻验收）

晚阶段8T混合背景验证中，GEMM阶段MAPE两场0.502%/0.248%，但六expert完成低估135..212us。
worker trace顺序替换表明入口第一场晚约35..59us、第二场基本对齐，同时gather服务存在重复的
基线低估与联合增量。因此不以GEMM竞争系数吸收启动误差，先固定无竞争gather。

令原逐worker服务为g0_j(M)，P(M)为真实gather panel数，S_j为worker_features的segments。
限定8T/H4096/M1..1205，候选为
`g_j(M) = g0_j(M) + beta * 1[P(M)>=8] * max(S_j(M)-1, 0)`。
P来自原native分工镜像：main=floor(M/12)+1[M%12>=9]，P=main+1[M-12*main>0]。
只在完整panel分配分支施加额外项；beta>=0，仅用large8_baseline/train/session1逐worker
中位服务残差最小二乘标定，beta=3.887047182501948us。它是经验服务修正，不宣称物理panel
延迟或独立read/write带宽。M1..84特征为零，原模型不变；正服务检查覆盖M1..1205。

已有18个未参与拟合形状的回顾检查：原12点两场worker MAE21.081/19.688us降至6.845/6.146us；
额外6点25.405/24.872us降至7.647/7.290us。测得worker entry条件下finish MAE也下降，
但M104/M106逐worker MAE两场均恶化约0.49us，尾/分工边界残差仍存在。
这些GEMM留出数据此前已采集，不是新gather前瞻证据；不接入默认planner、不改变竞争、entry或
资源需求。先冻结候选、补新gather验证，再决定是否采用。产物gather_large_candidate/model.json、
results.json、identity.json，完整协议与失败形状见joint_cost_model_20260911.md。

gather候选新增12形状前瞻验证已采前冻结，seeds611961/611962，门槛与逐worker数值见
`gather_large_validation/frozen_predictions.json`。两场串行采集中，尚未形成采用结论。

gather新12形状两场前瞻均通过采前门槛，支持保留隔离服务候选；第二场大M worker MAPE3.055%、
条件finish MAE8.857us。仍不代表自主计划或并发服务通过。

独立Lab gather动态响应以各worker无竞争service为工作量，r_j=(read+write KiB)/g_j，
`dx_j/dt=1/(1+alpha*sum_{active k, expert(k)!=expert(j)} r_k)`，按预测完成事件更新集合。
同expert已有并行成本包含在g_j；不额外施压。只用late_background/session1/n1代表性向量
拟合alpha=0.00175，其他条件用实测starts作诊断。n2/n4条件finish MAE改善，但n4 M529
高估约23us、M1205低估50..73us，平均bias抵消，不接入planner。名义logical需求不是物理
DRAM量，跨域、GEMM并发、依赖激活仍未覆盖。产物gather_response_candidate，公式仅为待证候选。

gather跨阶段诊断发现大M gather与较短peer W13大量重叠。固定原alpha直接叠加外生
W13名义8MiB/161.18us请求pulse虽改善M1205，却增加其他expert高估，n4 MAE两场
28.26/21.43→33.73/22.87us，拒绝采用。实测W13 starts仅为oracle输入，没有自主或反向反馈。
需独立辨识gather-gather与gather-W13后再验证统一时间线；不把该负结果改写为已解决。

分离n1/session1前台标定得到gather系数0.0016和W13 pulse系数0.0002，但n4 M1205仍
低估49..72us，M529高估约19us。两个系数尚未解决形状响应，保留split_results.json反例，
不部署、不把实测start条件诊断改称自主模型。

固定M1205前台、三组四个8T后台的同/跨LLC对照两场均显示约86..117us gather worker
服务增量，而同组local-cross仅数us。拒绝将该主效应仅局限同LLC；不唯一认定DRAM，
不据发现网格冻结共享容量。跨LLC共享域和阶段需求须进入下一验证范围。证据gather_shapes。

root-cohort自主事件原型已用预测gather结束触发W13/W2，不再输入实测stage starts。
但GEMM仍固定T0、没有反向反馈；沿用旧alpha/beta在新背景形状下gather服务低估60..107us。
仅事件顺序和无竞争不变量通过，需求响应迁移失败，不接入planner。产物unified_gather_candidate。

自主gather候选只用small/local四后台第一场配对增量重标定beta=0.0037，固定alpha=0.0016。
中/较大后台回顾误差仍约9..20us；1/2后台与cross2新预测已冻结并串行采集中，
尚未验收，固定GEMM/no reverse-feedback限制保持。产物unified_gather_recalibrated与gather_count_holdout。

1/2后台前瞻两场均系统高估减速，delta MAE23.006/20.519us，拒绝数量泛化。
保持固定窗口/系数改用实测stage starts的诊断只改变预测减速不足0.4us，不能修复主偏差；
该oracle存在阶段因果不一致，非可采用模型。需独立服务/非线性压力证据，不以同批失败数据
事后加knee宣称前瞻通过。证据gather_count_holdout。

独立1/2/4x8T阶段PMU两场+控制完成：流量可重复、计时扰动<0.60%，但同步吞吐不能
识别硬容量。32副本M48/W13每调用约15.4..16.4MiB，对照M12约7.5..8.0MiB，
不支持所有M统一Q8MiB；协议差异使其不能直接替代真实短时Q。保留证据、不采用容量。
产物independent_stage_service/analysis.json和reuse_contrasts.json。

独立team的命令占用PMU两场显示：W13/M12占用/读命令约36→45→80（1/2/4team），
即使吞吐近线性，代理仍显著非线性增长。其他stage/M对应关系不同，不能仅按GB/s预测队列。
这是吞吐/延迟分离的独立证据，不是实际gather加载延迟或计划未来PMU输入；需真实短时迁移验证。
queue_analysis_session1/2保留原始与逐controller指标，当前不冻结新模型参数。

独立稳态占用代理尝试q(n)=a+b*n²，仅用同shape第一场n1/n4标定；n2回顾MAE
0.390/1.073，小于线性count插值3.278/3.259。只限同质8T/M12或48/n1..4，不是异构队列模型。
新n3预测已冻结，queue_count3正在构建烟测，尚无前瞻验收，不能直接换算gather延迟。

新n3占用代理验证MAPE2.166%/2.465%，但同场W13/M48 n4控制第一场预测高17.10%、
第二场恢复，W2/M48 n4第二场高7.44%。保留控制失败，不能宣称曲线稳定或可迁移到瞬时异构计划；
n3局部插值结果不替代完整可靠性。证据queue_count3/evaluation_session1/2。

直接hybrid gather独立持续PMU两场完成：大M32输入/输出槽有显著读写流量，M1205约
35GB/s读与17GB/s写，每调用约19.2MiB读/9.5MiB写；源大小约9.4MiB，写分配解释仅为推断。
不能只建只读压力，也不能把同时轮换输入输出的数据外推成统一字节放大系数。真实T0保留，
需分开输入/输出复用并验证短时迁移。产物gather_native_probe/matched_analysis_session1/2。

直接gather输入/输出槽2×2两场均出现正服务交互（M529约14..18us、M1205约11..15us），
而每调用读写流量交互为负。不能用固定可加逻辑字节成本替代完整服务响应；也不据此追加经验
interaction项。下一步需可控GEMM后台下的前台需求/服务验证。证据gather_io_states。

直接gather+持续M12/W13后台的两场PMU显示：M529/M1205相对减速在n1/n2/n4均约4%/14%/44%，
后台吞吐仅下降约1..3.6%，占用/命令代理同步非线性上升；n4单份B控制前台几乎不减速。
可支持分离资源响应与前台敏感度的下一标定，但仅稳态双进程/大M全panel范围，
不得把未来PMU直接作为planner输入。证据gather_with_gemm。

分层稳态gather候选固定真实T0，q(n)=43.47618+3.49334*n+3.66666*n²，
T=T0*(1+0.00605628*(q-q0))。资源仅用M529/s1/n0,1,4标定，敏感度仅用M529/s1/n4；
M1205和n2回顾误差多在3us内。仅限同质B32持续后台与大M529..1205，非CPU延迟/异构模型。
M768/n3预测已冻结，扩展采集器正在烟测；不接入planner，不替代瞬时和完整计划验收。
产物gather_queue_model及gather_queue_holdout。

稳态gather队列候选的两PMU留出评分已完成：active service MAPE1.803%/2.319%，
但delta MAE8.758/7.084us，且存在基线高估与竞争低估抵消。使用各M同轮实际占用代理差，
固定T0/k的条件诊断MAE为8.481/4.545us；M1205/n3改善至-3.870/-2.412us，
M768/n2反而为-14.401/-9.786us。代理误差和敏感度/状态残差必须分开，不能仅修q(n)。
实测压力仅用于诊断；不作为自主预测输入，不把同质count映射用于异构或瞬时计划。
证据gather_queue_holdout/evaluation_session2.json及measured_pressure_diagnostic.json；
保持受限稳态对照，完整时间线与planner验收仍待完成。

完整anchor两场各31pair观测依赖路径均为42个16T任务，最后expert167；
最后W2中位数28697.71/28642.22us，不能用8T小前缀完成精度代表完整模型。
16T exact-M独立参照仅覆盖其中29任务，缺M14/16/18/20/22/26/32/43/73/77。
M1718联合gather比独立包络多约470us，后续小M同时存在入口包络差异与GEMM服务残差；
独立/联合CPU位置及expert上下文有差别，不作唯一竞争归因。
下一步优先16T服务/入口与缺失形状，同时保留全部8T背景反馈。
证据full_timeline_audit/results.json、baseline_transfer.json；这是观测分解，不是自主预测。

16T/M1718 worker诊断：联合平均worker gather858.943/857.857us，对比旧独立
412.087/418.602us；入口散布联合60.66/61.89us、独立64.11/66.54us。
保持联合实测起点并替换为独立local-worker服务，仍低估完成473.29/462.53us。
入口散布不是主要解释，但位置/状态/竞争仍须分离，不能将该差值直接当共享资源参数。
width16_anchor_missing已启动实际CPU240..255缺失十形状和M12/M1718位置控制标定；
参数冻结，不作为泛化留出。证据full_timeline_audit/gather16_workers.json。

完整224expert零竞争消融已闭合（full_baseline_candidate）：8T冻结解析服务、16T相同M
session1实测GEMM及worker gather服务，未知16T形状拒绝；初始入口使用历史31场景，
真实依赖/CPU释放按事件端点推进，无新trace时间作为预测输入。8T发布/setup参数在16T
仍属未验证迁移。完成27148.855us vs两场28697.71/28642.22us，低估5.397%/5.214%。
16T lane均值误差分解gather约-612/-620us、W13 -365/-243us、W2 -499/-535us、
入口/间隙-61/-119us。九lane均低估，需接双向动态响应，不按全计划残差加宽度倍率。
这只是完整DAG零响应对照，不是最终模型或泛化验收；16T全M、混合竞争与planner仍待验证。

完整动态事件消融full_dynamic_candidate将gather/GEMM请求窗口放进统一DAG，
lambda_i=d_i*v_i，v_i=1/(1+sum_type(a_type_to_receiver*sum_other_expert(lambda)))，
阻尼固定点收敛后推进到下个预测事件。通知/CPU释放只由预测W2完成触发；无实测端点输入。
31场景zero端点与旧完整基线误差<1e-7us，非零四组最多32轮收敛。当前仍用首块Q8/4MiB、
后续零共享需求及未验证的8T系数迁移，不是最终状态需求模型或硬容量模型。
全关闭/仅gather/仅GEMM/同时响应完成27148.855/27218.159/27414.505/27477.503us，
both仍低估两场4.252%/4.066%，不通过采用。需分段状态需求与敏感度校准，不乘全局残差倍率。
证据full_dynamic_candidate/results.json和ablation_summary.json，默认planner未改变。

显式分段接口segmented_dynamic_candidate将每阶段表示为(state,service_us,read_kib,
write_kib,sensitivity)序列；正服务成本之和严格还原冻结T0，非负请求随进度积分并逐资源守恒。
旧首块/private适配后四组×31场景全部224任务端点差<3e-10us，属于等价实现，不增加精度声明。
当前read/write仍合并进入issuer型响应，不是已识别的独立内存资源模型。
旧8T需求表只与anchor29任务/225路由行形状重合，真实状态/阶段时间分布未知，明确标为未测，
不得自动填零或从不同M整阶段请求差构造物理panel需求。
证据segmented_dynamic_candidate/legacy_parity.json和demand_inventory.json；需补16T/大M需求迁移。

16T大M整阶段需求标定large16_stage_demand已启动：M12/48/1718，W13/W2，B1/B32，
16T真实full stripes，synthetic连续routes，独立常数A/B，不等同真实4-copy/scrub状态。
M1718观测500ms、其余100ms，完成边沿计数并报告量化误差；首版packed-A尾部填充不足
导致烟测失败，v2补物理行后26格无PMU/PMU烟测通过，旧失败证据保留。正式三场串行采集中。
只采整阶段预算，不假设已经测得段内分布，不更新模型参数。8T大M及真实状态迁移仍待验证。

16T整阶段需求首场显示M1718/B32 W13读49.385MiB/call、W2读17.956/写9.513MiB/call；
B1大M仍有W13读27.123、W2读6.573/写6.814MiB/call，不能以B热推断后续总共享需求为零。
这是连续常数A/B探针、idle-subtracted/完成边沿归一化的整阶段预算，非panel或地址归因；
近零负write估计保留，不作为负物理请求。第二场重复与真实路径迁移尚待完成。
证据large16_stage_demand/analysis_session1.json和contrasts_session1.json，当前不更新模型参数。

16T整阶段需求两场完成：W13/M1718读预算相差约2%，但W2/B32读/写相差14.87%/27.44%，
B1相差26.44%/33.62%，而时长几乎不变；各场内部MAD小且前后轮次接近。
因此大M非零请求结论重复，但不能把绝对预算直接冻结为可靠形状函数，状态/分配因素尚未识别。
保留repeat.json和w2_drift.json，近零负估计不作为负物理请求。8T同协议版本26格烟测通过，
正式control+两PMU串行采集已启动；完整模型需求迁移和泛化仍待验收。

单独expert96/M1718的请求分布/接收mask消融（request_distribution_diagnostic）：
两场B32整阶段预算×前半/均匀/后半发出×首窗口/全阶段接收，12×31完整DAG场景均通过守恒。
该受限设置三种分布完成跨度<0.15us，不是通用时序不敏感结论。全阶段相同响应使完成增加
约407us，却将W13从低估61..89us变为高估259..288us，与gather低估408us抵消。
不采用全阶段统一敏感度，需独立标定输入/阶段/历史相关接收响应；不据总误差改善加倍率。
这是回顾诊断，未证明需求状态迁移或planner泛化。

8T/M1205整阶段需求两场完成：B32 W13读39.5745/39.5503MiB，W2读16.0585/17.1721、
写8.1993/8.9449MiB；B1 W2读2.8949/3.6787、写3.5999/4.7583，状态偏移并非仅16T。
不直接采用固定预算或由整阶段推断panel流量。width16_response已启动真实路径标定：
固定16T M77前缀、16T前台M12/48/1718，对照0/local1/local3/cross3个长8T背景。
需用trace验证实际阶段重叠并分离前缀时序变化，才可拟合接收敏感度；这批不是泛化留出。

width16_response两场完成：前台M12/M48与三长背景W13近100%重叠但减速仅约0..3us；
M1718 local3 W13/W2增量12.23/8.84和17.78/8.51us。背景已运行一段时间，非完整anchor的
并发/阶段变化条件；保留为弱响应对照，不用近零增量标定强响应。
width16_wave_response进一步以三条实际21expert小/长/小队列创造阶段变化，原路由/几何保持，
两场标定采集已启动。目标gather的少量负增量也须与共享输入/状态区分，不盲加正倍率。

receiver16_candidate预划分M12/M48 local1训练后，first/reuse拟合W13=.90/0、W2=.35/0，
但7个未拟合条件的增量MAE两场由6.965/8.497变为7.798/9.330us，拒绝采用。
实测起点/背景时序诊断仍为7.631/9.120us，不能仅修时序。M1718/W2 local3增量29.94/49.08us，
cross3为-3.47/3.67us，而名义首窗口重叠量两种放置均约281..284MiB。
gather local/cross均减速约183..196us；下一版需区分跨LLC共享与本地共享域响应，并检验非线性，
不通过team倍率弥补。以上为真实队列实验及有限名义需求诊断，非唯一硬件因果或最终模型。

local_domain_candidate增加按既有40核LLC分组的同域接收项：
v=1/(1+s_global*P_global+s_local*P_same_domain)，请求量仍只生成/积分一次。
local默认0，124个完整DAG回放与原模型端点差<3e-10us；隔离和跨域压力不产生本地减速。
已看过的M1718/local3 W2标定得到reuse local系数0.28，跨域和W13预测保持不变。
除标定条件的W2增量MAE两场2.910/2.810us（初始2.375/2.883），绝对阶段MAE有所降低，
但存在噪声和分项抵消边界，不宣称全面改进；需新的数量/形状冻结验证，再看完整模型。
本项是有效局部资源响应，不是宽度奖励/惩罚，也不是已证明的LLC缓存缺失模型。

本地W2 reuse0.28的前瞻验证width16_local_holdout已采前冻结：M12/24/96/1718，
none/local2/cross2，共12对照+anchor，双模型各31历史入口场景，不读取新实测时序。
本地项预测local2额外W2为M24 .606us、M96 3.908us、M1718 17.139us，M12与cross不变。
真实两场采集已启动，结果未出；按绝对阶段误差、n0配对增量和local-cross差/MAD评分，
不以此替代完整模型/搜索验收。

本地项前瞻两场完成：W2阶段MAE由4.633/5.843降到2.745/3.813us，local-cross差MAE
由3.360/6.358降到3.182/1.552us；M1718差值噪声较大，支持保留组件而非精确物理系数声明。
full_local_transfer对完整anchor回放，family8+local16完成27482.750us，仍低估4.234%/4.048%，
local项只增加6.770us，完整目标尚未闭合。后续段零共享需求仍是待解决的结构缺口。
硬件只读核对L2=1280KiB、LLC=71680KiB/40核域；B owner大小不等于命中率或无后续请求证明。

后续段需求的L2/DDRC分离观测已启动l2_stage_demand16：内核别名确认PMU type8，
L2 access/refill/writeback事件0x16/17/18，16核48事件与64DDRC合计112事件烟测通过。
复用原验证二进制，整阶段每调用事件数保留，不把refill直接等同精确LLC字节或纯B请求。
正式控制+两PMU串行采集中；目标是约束后续共享需求，尚无新数据结论或参数更新。

L2/DDRC首场：16T/B32 M12→M48，W13 refill增长4.10倍而DRAM read约2.03倍，
W2分别3.30/2.11倍；M1718/B1仍有W13约387万、W2约81万refill/call。
支持分别保留本地请求与DRAM需求观测，但system-wide事件含运行时且无panel时间归因，
不换算精确LLC字节。第二场继续验证；8T/88事件版本仅准备，参数和默认行为不变。

16T L2/DDRC两场完成，B32 M12→M48的refill约4.1/3.3倍而DRAM读约2倍，重复支持
分层需求特征；但部分B1中M及大M事件仍跨场变化，不能冻结一个无状态精确请求表。
8T对应88事件烟测已通过并开始串行正式采集，复用原二进制；尚未更新模型参数。

request_budget_bank提供有单位、来源和观测场次的整阶段向量，16T两场12格已整理；
L2事件不转成精确LLC字节，负idle估计保留，未知M/段内时间分布不填零，真实路径迁移标记未通过。
8T首场88事件完成，M1205/B32 refill约W13 1078万/W2 159万；第二场仍在测量。
该库是拟合输入的观测记录，不是已采用的时间模型或固定无状态请求预测器。

### 2026-09-12 Lab gather worker响应回顾对照

冻结独立worker gather成本、legacy请求、入口与其他系数，只标定
`gemm_to_gather`，以M1718/16T、一个8T混合队列的配对worker平均执行减速为目标。
不使用阶段包络或背景任务数作为响应目标/特征。候选系数0.00176953125对比原0.0002，
其他51条已见记录的增量MAE22.426→6.883us，绝对worker服务MAE21.302→8.074us。
三队列预测约122us仍低于175–185us实测；保持后续请求为零的旧表达也未被验证。
因此仅保留为回顾性结构对照，不能当作独立需求模型、硬件参数或新鲜泛化证据。
模拟器增加worker端点输出，在训练计划剥离新增字段后完整结果与原engine精确相同；
事件方程未改变。完整计划迁移/强压力非线性/新数据验证仍未完成，默认模型不变。
记录与数据位于`results/joint_cost_model_20260911.md`和
`tmp/joint_cost_model_20260911/gather_response_worker/`。

### 2026-09-12 Lab gather请求率二次响应

在冻结需求、独立成本和其他响应参数下，仅gather接收端使用
`v = 1 / (1 + c_GG*q_G + a*q_K + b*(q_K/100)^2)`，其中`q_K`是其他expert
当时实际GEMM发出率（KiB/us，随执行减速反馈），`q_G`为对应gather发出率。
100 KiB/us只是数值归一化，不是硬件峰值；没有加入队列数特征。b=0恢复线性模型。
参数非负，沿用原反馈迭代及不收敛拒绝；不改变请求预算。常速发出方解析时间、独立极限、
负/NaN参数拒绝与两训练计划b=0完整输出相等检查通过。
已见wave/session1 M1718 local n1/n3回顾标定a=.001220703125、b=.083203125。
其余50条既有记录增量MAE线性5.817→二次2.583us，但绝对服务MAE7.499→8.596us。
两队列集合增量MAE3.706→3.778us、绝对MAE3.458→9.380us，不能以总增量指标掩盖恶化。
标定仅覆盖16T接收方，完整回放应用到其他宽度属于待验证迁移；后续零需求等旧表达缺陷
仍保留。默认不采用，完整计划回放及新的条件验证未完成。详见同日报告与
`tmp/joint_cost_model_20260911/gather_response_quadratic/`。

二次gather候选的原224task anchor回顾回放已完成：原/线性/二次完成时间
27482.750/27665.478/27803.196us；二次对原两场误差-3.117/-2.929%，仍低估839–895us。
原臂31入口场景全部端点与既有full_local_transfer在1e-7us内一致。M1718 gather仍低估，
其他team宽度响应迁移、后续需求、新计划和planner搜索仍未验证。固定无竞争gather均值
416.256us与前缀条件n0的392–395us差异独立保留，不能重新拟合竞争去维持误差抵消。

2026-09-12新队列顺序前瞻验证结论：冻结二次响应不推广。保持expert集合/team数相同，
frontloaded/staggered两种新顺序在两场均使二次增量MAE劣于线性：M1718线性9.129/6.897us，
二次14.981/14.388us；M96线性1.090/1.178us，二次1.668/1.756us。冻结比较门槛未通过。
原完整anchor回顾误差下降不能替代此验证。线性仅保留为对照，不自动采用；不在本holdout
继续调参，优先审计需求时序与前台敏感度。详见gather_response_holdout/decision.json。

### 2026-09-12 Lab自身基线归一化资源响应

回顾诊断`h(P)=1+aP+bP²`，`P=(R+wW)/100`（R/W为GB/s），相对响应
`h(P_joint)/h(P_solo)-1`保证相同状态变化为0。参数a=.01、b=.10、w=2来自有限网格，
w落在边界，不解释为硬件常量。实测压力oracle的34非训练格MAE1.768us，简单外部对照2.316us。
只输入solo数据、按前台`cycle(v)=max_worker_T0/v+overhead0`推导请求率反馈后，MAE2.739us，
尚未优于对照。背景速率仍固定，worker统一速度，不能代表完整资源反馈或任意计划预测。
联合流量不作为正式预测输入；无背景v=1极限通过。下一步检验背景反馈，默认不采用。
记录见`gather_self_pressure/oracle.json`与`solo_prediction.json`及同日报告。

双角色稳态反馈后续：保持h固定，背景`vB=1/(1+gamma*max(0,h(P)/h(PB0)-1))`，
前台`vF=min(1,h(PF0)/h(P))`与原cycle关系一起求总P；gamma=.121234566由一个既有训练
条件的BG吞吐标定。gamma0旧solver一致、无背景/对称极限和输入拒绝通过。
34非训练格时间MAE2.312us，但总read MAE2.663GB/s未改善，仍不推广。
以双方实测调用率重构联合流量仍有状态相关残差，固定solo每调用下层请求预算尚未成立；
需求状态与服务反馈需分开验证。详见gather_self_pressure/coupled_prediction.json与budget_audit.json。

2026-09-12热参考差额消融：保持h/gamma不变，逐worker仅放大
`max(0,T0-Thot)`，保留原T0与cycle余项。无背景/热reference不变检查通过；IO1/1 MAE
12.124→.338us，但32/32为1.924→18.516us，不推广。热组零响应由构造保证；未重拟合
参数和热冷差的非物理可分性限制归因。下一步需区分重叠与新增需求，不能把差额当纯访存成本。
见gather_self_pressure/exposed_prediction.json。

### 2026-09-13 Lab请求服务重叠候选

固定各worker T0，令`L_i=min(T_i0,kr*Qri+kw*Qwi)`，
`T_i=max(T_i0,L_i*max(1,h(P)/h(P0)))`；cycle取最慢worker加原余项，保留BG反馈。
Q为solo每调用总流量按逻辑工作份额分配，尚非逐worker实测需求；kr/kw不是硬件带宽。
三格回顾标定kr=1200/kw=200 us/MiB，33非训练格增量MAE1.216us，整体放大/差额版
同集合5.462/3.927us；总read约2.08GB/s、write约.97GB/s误差未明显改善。
无背景和零latent极限、份额守恒检查通过，候选仅保留，不采用。中间复用状态、真实计划、
需求变化和完整planner验收仍缺失。见gather_overlap/results.json及同日报告。

2026-09-13中间IO前瞻验证：solo profile后、joint采集前冻结的重叠预测，M96两场
MAE .970/1.019us、max2.588/2.655us通过；M180 MAE2.213/2.217us、max6.043/6.142us失败。
16/1实际增量近0而预测5.850us是主要缺口。比全成本对照改善不等于通用模型通过；
`gather_overlap_holdout/decision.json`明确not_promoted，参数未回调，默认保持。

### 2026-09-13 Lab：固定预算的阶段时序／响应分离诊断

`tmp/joint_cost_model_20260911/plan_completion_audit/ablate_segments.py`
在完整历史计划评分之后，使用两个既有计划做2×2机制对照，不改变生产模型或planner。
每个stage的独立服务分段为 \(c_b\)，请求总预算为 \(Q_r\)。请求时序因子比较原首段集中
与 \(q_{b,r}=Q_r c_b/\sum_b c_b\)；二者保持 \(\sum_b q_{b,r}=Q_r\) 和全部 \(c_b\) 不变。
响应因子仅比较8T后段原有零敏感度与该stage首段敏感度；16T响应和全部拟合系数保持冻结。
四个条件须复现相同的零竞争端点，原条件须保持历史预测；不使用实测joint端点作为输入。
这不是从PMU测得的panel时序，也不声称首段敏感度可迁移至后段。已有B32全stage请求量
仅提示首块预算不足，尚未验证真实路径迁移；本对照故意保持旧总预算，以分离时序与响应。
结果仅决定后续独立标定方向，不用于在该历史集合调参后宣称泛化通过。完整历史集合评分已完成，
2×2计算对照已完成：anchor时序单因子约+4.98us、后段响应约+94.51us；control_forward
分别约-.68us、+486.10us。后者的8T/W13和W2路径低估减少，但仍有残差，不能由此采用
统一敏感度或证明请求时序无关。下一步须在独立可控竞争条件中标定后段响应，再验证动态
迁移；当前全部结果只作历史机制诊断。记录见 `optimizations/fused_moe_sve/results/joint_cost_model_20260911.md`。

### 2026-09-13 Lab：共享逻辑请求容量诊断

`plan_completion_audit/capacity_simulate.py`是冻结event simulator的独立实验快照。
保留原响应固定点生成的速度 \(\widetilde v_i\)，对当前发出请求的活动片段使用
\(R=\sum_i r_i\widetilde v_i\)、\(\alpha=\min(1,C/R)\)，将其速度改为
\(v_i=\alpha\widetilde v_i\)；零请求片段不受这项缩放。原压力反馈继续迭代，并在每个
事件区间检查 \(\sum_i r_i v_i\le C\)。未传容量时保持旧行为。
首轮诊断固定 \(C=200000/1024\) KiB/us，不拟合。\(r_i\)仍来自旧GEMM/gather逻辑预算，
所以C是有效逻辑服务率，不是已经验证的DRAM带宽；不能将所有逻辑请求当作实际DRAM流量。
检查覆盖1/3/5个同步team的 \(\max(T_0,nQ/C)\)、关闭/高容量时完整anchor端点等价，
以及非法容量拒绝。随后只在已有M12/13/48可控cohort比较容量开关及后段响应开关。
该机制诊断不改变生产模型、默认planner或已冻结的独立成本，不构成新鲜泛化验收。
固定200诊断已完成：可控M12/M13的W13增量MAE从9.429/12.362us降至3.148/1.775us，
但M48从6.829升至7.379us。完整anchor/control_forward预测29261.321/30022.113us，
相对两场实测高估约1.96..2.16%/.83..1.25%。同时开放全部8T后段响应使这两计划高估更大。
因此不能独立采用前一轮后段系数，也不能以总时间改善证明请求预算/阶段映射正确；
后续容量须从独立竞争条件标定，并保持完整计划只用于迁移检查。
随后仅用历史M12/cross_n4/session1的W13配对增量38.48us，固定其它参数，在150..300
等效GB/s区间作16轮二分标定，得到206.86455。该值是开发标定，不是物理带宽测量。
冻结后历史anchor/control_forward误差为+1.36..1.56%/+.19...61%；其余M的响应仍有残差。
新M17/M35联合条件已准备，须先冻结全部数值预测再测量；在独立验证前不采用此参数。
新M验证随后完成：四场各369call校验通过，M17/M35的W13配对增量MAE分别
1.434/1.733us和3.916/3.571us，W2分别.930/.648us和.622/.840us；各场各stage均通过
预设MAE5us、max12us、相对原模型MAE回归不超过1us门槛。结论仅覆盖同路由/同背景的
新joint M，不能扩展为新工作负载、所有宽度或planner改善。默认模型保持，继续完整计划迁移。

### 2026-09-13 Lab：uniformish完整计划迁移范围

冻结206.86455等效容量在另一route/layer的234-expert、全8T、四个固定lane顺序上完成
两场采前预测验证。完整时间MAE393/239us，对照原Lab响应1629/1681us；每plan完成时间
误差<=5%、十lane误差<=max(100us,10%)及候选集regret<=2%的预设门槛均通过。
两模型都选anchor，所以此结果不证明搜索选择改善。small_first的实测配对减速稳定，但
容量模型夸大幅度，仍保留误差。当前provider仅支持这234任务的全部8T成本、155个16T成本，
1/2/4T无实现；完整混合宽度搜索须先补齐并验证成本，不能将全8T顺序对照当作该目标完成。

### 2026-09-13 Lab：窄team gather打包路径候选

`width_provider_audit/gather_paths.py`对每个实际gather工作片段使用
\(G=a_p+b_p Q_{\rm read,KiB}+c_p Q_{\rm write,KiB}\)，\(p\in\{8,12\}\)为packed rows。
系数跨1/2/4T共享；到达时序不在此worker服务公式中。用同expert M1..12第一场奇数M作
非负最小二乘，每个(width,M)条件总权重相等。8行路径系数约(1.41679,.105715,.079857)，
12行路径(1.708158,.417754,0)。这些是有效回归系数，零write系数不表示写入没有成本。
已经看过的偶数M两场条件等权MAE为.688/.778us，对照不分路径3.132/3.160us；该划分
是开发检查，不是前瞻验证，分路径模型也使用更多参数。
默认只接受M1..12；显式实验选项可将各片段服务相加到M96，以待测新M验证。单panel数据
不能区分每片段启动开销和每worker一次启动开销，因此不能提前采用多panel相加假设。
已冻结M13/15/17/19/21/23/24/36/48/60/96的worker预测，M12作控制；GEMM另作基线标定。

### 2026-09-13 Lab：窄team多panel验证与GEMM候选边界

冻结gather分路径模型在1/2/4T六场均通过预设相对MAE10%及每worker max(3us,15%)门槛。
1T误差随panel数增加且绝对误差高于shared对照，仍保留；shared对照在2T/4T未通过。
结论仅覆盖已声明的同expert、M<=96选定多panel组合，不是联合运行或任意大M验证。
`width_provider_audit/narrow_gemm.py`另定义待验证候选：M<=12保持首块实测成本；
M=12h+r>12时，\(T_s=T_s(12)+(h-1)b_s+d_{s,2\lceil r/2\rceil}\)，r=0不加尾项。
\(b_s\)只用第一场完整块点拟合，\(d\)只取h1奇数尾块配对差。第二场最大相对误差
约2.20%，但这仅是已测形状重复；偶数尾块、后续历史及M>96都没有通过该候选验证。
1T低M首块状态漂移不因上述重复结果而消失，不能据此直接补齐完整planner成本接口。

### 2026-09-13 Lab：窄team独立组合／大M验证准备

不改变M<=96候选或已有参数，单独的`narrow_extrapolation.py`将同公式用于至M1718的
明确待验证预测。保留M12控制，验证M14/18/22的偶数h1尾块、M25/35和49/59的h2/h4尾块，
以及M384/697/1200/1718。数值正值/有限检查不是准确性证明。
GEMM预设每stage条件等权相对MAE<=3%、每点误差<=max(5us,5%)；gather单独要求条件
等权相对MAE<=10%、每worker误差<=max(3us,15%)，M12控制单列。三宽度正确性烟测通过，
预测已冻结，六场正式采集已启动；当前没有扩大已验证的provider范围或改变默认planner。
随后六场完成：gather和W13均通过冻结门槛，W2六场均未通过，M49是重复失败点，2/4T还
有大M高估。保持原失败记录，不采用完整窄team provider。M384/1200完整块的观测差商
乘width约575..576us，而短范围later-full常数归一化后约593/605/614us，说明W2尚需
区分渐变与后期成本；该差商是诊断，不是硬件峰值或已冻结的新参数。

### 2026-09-13 Lab：W2暖态完整块与历史净尾项候选

保留前次失败结果后，仅用第一场开发数据修W2。令 \(b_T=575.51/T\)，\(c_T=T^0_2(12,T)\)，
完整块前缀采用 \(P_T(h)=h b_T+(c_T-b_T)\sum_{j=0}^{h-1}\rho_T^j\)。
对tail kernel rows \(k=2\lceil r/2\rceil\)，令 \(u=(k-2)/10\)，净尾项为
\(d_{T,k}(1)+[(1-u)A_{T,2}+uA_{T,12}](1-\eta_T^{h-1})\)。
\(\rho\)只拟合完整块，\(A,\eta\)只拟合已测端点kernel的后续历史；中间类别插值待新点验证。
该尾项是whole-shape净变化，不能解释为被单独计时的物理尾块服务；kernel2幅度触及
\(-d(1)\)边界也不表示实际计算免费。W13、gather、首M1..12及h1尾项保持不变。
模型文件`width_provider_audit/w2_warm_candidate.json`；旧第二场重复误差改善不能挽回
原冻结验证失败。新M27/29/31/33、51/53/55/57、37、192/768已冻结预测，M12作控制，
六场正式验证已启动；未采用到完整混合宽度provider或默认planner。
随后六场结束：1T/2T新点全部通过，4T第一场M31高估5.061%而第二场4.704%，原门槛
保持且整体记为not_fully_passed。4T/M55两场仍高估约29.59/28.35us，说明中间kernel
历史插值有重复残差，不只是一项通过率统计。gather/W13及控制全部通过，下一修正仍只针对W2。

### 2026-09-13 Lab：W2中间kernel独立历史节点

`width_provider_audit/w2_kernel_history.py`仅替换kernel rows4/6/8/10的净尾项。
保持前版完整前缀P_T(h)、h1节点和kernel2/12端点；h2/h4节点用前次transfer第一场
whole-stage中位数减固定P_T(h)得到。h1到h2、h2到h4分段线性插值，h>=4保持h4节点。
这是待验证的常数延拓假设，节点不等于单独计时的物理尾块服务。前次transfer数据现在
属于开发数据，旧not_fully_passed结论保留；第二场开发重复相对MAE约.361/.330/.316%。
5154个M/width检查保持W13、首块、h1和端点行为且输出正值有限；不构成准确性证据。
新冻结验证为M39/41/43/45(h3)、63/65/67/69(h5)、199/775(长历史kernel8)，M71端点
和M12控制；1/2/4T各两场。沿用此前分项门槛，当前仅完成本地准备，未开始硬件采集。
该变更只影响Lab候选，不改变默认planner、生产成本接口或Plan V2。
随后六场冻结验证全部通过原分项和控制门槛；W2相对MAE1T .308/.472%、2T .630/.588%、
4T .531/.528%。4T同点旧warm公式绝对MAE18.671/18.549us，节点版9.017/9.018us；
2T接近持平。decision为passed_declared_kernel_history_gate，仅声明同expert独立测试域。
16T、混合竞争和完整搜索尚未验证，旧失败记录保留。

### 2026-09-13 Lab：16T完整块渐变候选

`width_provider_audit/fit_full16.py`只拟合16T full-stripe完整块，令
P_s(h)=h*b_s+(c_s-b_s)*(1-rho_s^h)/(1-rho_s)。c固定bulk第一场M12，
在rho=0,.001,...,.999网格上对第一场完整块作条件等权相对平方误差最小化；
每rho解析求b，约束0<b<=c。W13参数(c,b,rho)=(93.14,75.450350,.658)，
W2=(44.52,36.449660,.702)，单位us。这是假设性的有效服务收敛，不唯一归因为cache。
保持h1锚点，h1..143正值和单调性检查通过。第二场开发重复相对MAE W13 .617%、W2
1.018%，最大2.060/4.258%（W2最大在M12控制）；未采新holdout，不能宣称泛化通过。
未拟合尾块或gather，尤其不能用h140异常负净差作为物理尾块服务；原provider不变。

### 2026-09-13 Lab：16T绝对误差前缀与短历史尾项候选

保留relative-fit前缀，`fit_full16_absolute.py`使用相同参数量和第一场完整块，
改为绝对平方误差拟合。W13(c,b,rho)=(93.14,75.009457,.749)，W2=(44.52,36.289064,.769)。
第二场W13 MAE由9.276降至2.718us；W2由3.441升至3.891us，保留退化。
`fit_tail16.py`固定该前缀，用第一场h1/h2/h4的whole-stage减前缀得到六种kernel
净尾项节点，历史间线性插值、h4后常数延拓。M1..11保持first_rows第一场，M12统一
bulk前缀锚点。h140两场全部保留为开发诊断，不用于拟合节点；不能将节点当物理尾块。
3436个M/stage正值检查通过。短历史第二场最大误差W13/W2=3.948/3.471%；
长历史开发数据最大误差=.397/5.638%，W2包络异常仍未解释，不能宣布16T验证通过。
两候选均Lab开发范围，尚无新M验证、gather组合或联合环境支持，原provider不变。

### 2026-09-13 Lab：16T gather worker服务候选

`fit_gather16.py`采用G_w=a+b*n_segments+c*read_KiB+d*write_KiB，工作片段来自
实际M-panel/K-stripe镜像。第一场first_rows和bulk标定，bulk重复M12不重复计权；
非负最小二乘按max(3us,实测worker服务)归一化残差。矩阵rank4，系数为
(0,.234406782,.332427214,.133418525)，不是独立物理带宽；零截距不证明启动免费。
第二场first_rows MAE.173us/相对11.031%，bulk4.071us/3.499%；每worker均满足
max(3us,15%)但first_rows平均相对误差超过10%，不能宣称全部旧门槛通过。
模型仅预测worker服务，不包含到达跨度；M1..1718输出正值有限。新M和联合运行仍待验证。

### 2026-09-13 Lab：16T新M冻结验证准备

GEMM尾项候选与gather服务候选合并为分项预测，冻结M14/18/22、39/43、65/71、
192/1692、775/1201，M12控制。数据和参数不因评分回调；两场沿用GEMM相对MAE3%、
单点max(5us,5%)以及gather相对MAE10%、worker max(3us,15%)。此时仅本地准备完成。
新点通过不能替代旧h140完成时间异常解释、M1..11相对误差、联合需求或完整搜索验证。
随后两场结束，decision=not_fully_passed。gather两场M43/M65同worker高估，主要涉及
8行尾段；W2第二场M1692低估5.661%，包络增加约310us而worker平均服务中位数仅增加
13.43us，慢线程完成分布仍未解决。W13两场通过，不能将部分通过作为完整provider采用。

### 2026-09-13 Lab：16T gather分路径需求修正

`fit_gather16_paths.py`按packed8/12各自累计片段数、读KiB和写KiB，六系数非负拟合；
训练数据和损失保持，只用原第一场标定。旧新点已参与结构选择，复算属于开发诊断。
旧full16两场gather相对MAE4.073/4.090%，M43/M65及其他worker不再超单点门槛；
原bulk第二场绝对MAE4.071->4.599us退化保留。需要新的M验证，旧失败决策不撤销。
GEMM和长W2完成分布问题不因这一修正而改变，原provider不变。

### 2026-09-13 Lab：16T gather分路径新点冻结

固定分路径gather候选和原GEMM，冻结M26/32/38/44/62/68以及179/180/181/188/193，
M12控制；后组检验panel数跨过16时的工作分配变化。两场沿用原分项门槛，当前烟测中。
不重新拟合，不把新点结果作为长W2慢线程或真实联合需求验证的替代。
随后两场结束，gather M68/M188重复失败，M44第二场也失败，均余8；W13/W2通过。
decision=not_fully_passed。native m8循环无显式完整8行专用kernel分支，尚不能唯一
归因于kernel切换或缓存；有效源行数与K片段长度需要独立验证。

### 2026-09-13 Lab：16T有效源行配对诊断

固定work分配，比较base={0,36,60,180}各自M=base+{6,7,8}，逐pair计算
D7=G7-G6、D8=G8-G7以及D8-D7。原公式冻结，M8控制；含旧点，属于机制诊断。
尾段K长度从约240到4096元素变化，检查源行增量与片段长度关系；暂不新增模型系数。
work等价不等于缓存或指令状态相同，不能据此单独声明硬件原因。
随后两场完成，长片段D8-D7重复：K约1936..2160时第一场6.07..6.51us、第二场
6.22..6.40us；K4096为14.14/13.89us。支持为全部8行有效片段建立待验证服务候选，
不支持把D2无条件加到有偏基线或宣称跨输入泛化。

### 2026-09-13 Lab：完整8行gather服务候选

仅packed8且valid8片段改为G8full(K)=.056573513+7.298524055*K/1024 us；
partial8与12行片段保持原分路径系数。参数只拟合配对控制第一场22个tail worker，
减去混合worker中固定12行服务，采用max(3us,actual)归一化非负最小二乘。
第二场worker MAE.314186us/相对8.238088%，无单点超限，但只是同点重复。
M余数非8保持，全部预测正值有限；新的M与联合环境尚未验证，旧失败记录不撤销。

### 2026-09-13 Lab：完整8行服务新M冻结

固定valid8服务与其他路径，冻结M20/56/80/104/140/176/200/776，79/103/108作未改
路径guard，M12控制。两场沿用原GEMM/gather分项门槛；当前仅准备与烟测，无新点结论。
随后两场结束，valid8新M gather在M80/176/200重复高估，GEMM通过；decision未通过。
M200同一8x4096工作仅约14.42/15.44us，对比M188约28..29us，valid8/K长度特征不足。
下一步做固定工作量的输入位置对照，不将候选作为通用成本采用。

### 2026-09-13 Lab：固定M的源token位置干预

source_position16对M188/M200分别构造base、改变奇偶占比的单行交换、保持占比的
单行交换。每expert数量、target前M-8源行、CPU/工作量和同M输出位置不变；源值与地址
一起改变。比较配对worker gather时间，不增加cost系数；该干预不单独证明物理cache索引。
独立Lab runner保存旧快照，两场位级与输入tensor身份核验通过。
M188改变占比3/5->4/4配对变化-14.00/-13.03us，M200 4/4->5/3为+14.23/+14.02us；
同占比换行对照约-.46..+.38us。支持实际源地址类影响服务，但不证明唯一物理cache机制。
后续候选需要source-token信息或明确的不确定性；expert M计数本身不足以确定该状态。

### 2026-09-13 Lab：source-token条件gather候选

完整8行片段按实际source tokens奇偶最大占数c选择G_c(K)=a_c+b_c*K/1024。
c4系数(1.116485,3.384607)，c5=(.560325,6.988803)；仅当前16T/H4096布局开发候选，
不宣称物理cache-index证明。输入必须提供actual tokens，c>5明确未支持，不能从M猜测。
旧第二场MAE .407/1.004us，c4平均相对10.652%限制保留；无新holdout。
原expert96五个余8形状落入c>5，尚不能完整覆盖planner。partial8/12与GEMM保持。

### 2026-09-13 Lab：新源集合与更高占数采集准备

固定M212/M224，每个M构造尾8源行最大奇偶占数4..8。c4/5四个条件冻结gather预测，
c6..8六个条件仅采集且预测为空，不计入准确性通过率。保持目标前M-8源行与全部expert
数量；11plans、两场，使用原gather门槛。
随后两场完成，四个c4/c5新源条件相对MAE3.049/2.302%、单点均通过。c6..8服务在
两场重复，但全部K4096，只是discovery，不能宣称更高占数跨K支持或联合验证。

### 2026-09-13 Lab：高source占数增量候选

c6..8只新增G_c(K)=G_5(K)+.566549731*(c-5)*K/1024 us。eta由第一场六个K4096
条件拟合，第二场MAE.960799us。c4/c5及其他路径保持不变，跨K缩放尚未验证。
原expert96全部M可生成数值候选不等于已覆盖准确性，需独立验证后才能接入完整provider。

### 2026-09-13 Lab：高占数跨K冻结验证

固定M32/92/152，各构造占数4/5/6/8的新source集合，12条件全部冻结预测。尾K片段
覆盖56..4096元素，评分仍逐worker，避免长worker掩盖短片段错误。两场原gather门槛，
随后两场完成，12条件相对MAE3.522/3.481%、worker MAE.716/.671us，单点全通过。
仅保留为带source-token的Lab服务基线；不将GEMM未评分、任意源路由或整体联合运行视为通过。

### 2026-09-13 Lab：扩展独立成本接口

expanded_baseline.Baseline.service(M,T,source_tokens)整合窄team、16T候选与注入的旧8T
服务，返回gather逐worker和两个GEMM成本及限制。16T强制源tokens，8T M>1205仍拒绝。
8077组合与原候选逐项一致，仅为接线验证。成本未包含gather到达、联合竞争和慢线程
完成波动；限制字段不等于已标定置信区间。生产和旧provider未修改。

### 2026-09-13 Lab：8T大M外推验证准备

保留全部8T系数，在独立候选中将max_m扩至1718；原接口仍拒绝>1205。
expert96的9个M>1205点冻结预测，M12/1200/1205单独作为范围内guard；每stage均值
只计算9外推点，原门槛不变。
随后两场全部通过，第二场gather/W13/W2相对MAE1.191/.146/.162%，guard也通过。
Lab接口显式extended8 callback可用于1206..1718并检查通过decision；<=1205仍旧回调，
未配置时仍拒绝大M。1718个输出接线一致，不构成联合或任意route泛化证明。

### 2026-09-13 Lab：独立成本替换的完整历史回放

固定31arrival场景、全部请求/响应和容量参数，只替换独立service。anchor预测
29088.960->29068.013us，control_forward29832.188->29822.080us，误差小幅改善。
旧结果复现，放置/依赖/请求和响应字段逐项一致。仅8T/16T两计划历史诊断，不构成
窄team联合、后续需求正确性或等预算搜索结论；legacy后续零请求假设仍未修正。

### 2026-09-13 Lab：窄team持续背景联合验证准备

前台M48/T1,2,4，背景四个8T真实expert M1205/768/714/529，local/cross n0,1,2,4。
冻结隔离成本、arrival和原capacity/response，保留legacy后续零请求。四背景下模型只
预测W13约15..27us增量、W2零增量且local/cross几乎相同；用配对n0作诊断，
预设stage delta MAE5us/max12us。
随后六场结束，整体未通过原门槛；4T local n4 W13实际24.75/29.58us、cross15.02/
15.34us，局部差异较小但重复。W2没有显著持续减速，不能等价为后续物理请求为零。
四个8T背景不代表更多窄team，后续固定总核数改变team组合验证。

### 2026-09-13 Lab：固定背景核预算的team组合验证准备

固定4T/M48前台，背景32核分别32x1T/16x2T/8x4T/4x8T，local/cross加bg0。
实际背景expert集合随team数量变化，全部形状进入冻结模型；不是相同计算工作量的纯
width消融。旧模型预测W13约16us，W2在32x1T约22.7us、其余0..3.3us，局部差异小。
配对delta门槛保持。
随后两场结束，整体未通过。local16x2T W2实际38.15/28.83us而预测3.256us，cross
10.70/10.39us；局部差异重复但幅度有场次变化。下一步用独立窄team需求测量分离后台
压力和前台敏感度，不将任务数直接作为惩罚或假称物理LLC流量已知。

### 2026-09-13 Lab：窄team独立请求标定准备

standalone1/2/4T测M12/48/96、W13/W2、1/32copies；观察DDRC读写与core L2事件，
保留idle-subtracted结果和completion-count误差。独立服务成本不替换runtime T0，
不把L2事件直接当LLC字节。预设计数边界5%、PMU扰动5%、counter running99%质量限制。
当前仅源与协议准备，native未构建，尚无新需求数据或竞争公式变更。

### 2026-09-13 Lab：请求增量的分层统计

在独立需求协议内按同round计算每调用预算Q(M)，再比较[Q(48)-Q(12)]/3和
[Q(96)-Q(48)]/4。保留有符号差和端点质量标记，不能解释成直接instrumented panel流量。
1T第一场12cell测量质量通过；持续L2 refill与状态相关DRAM增量同时存在，支持将二者
作为不同单位的需求特征。重复场次/其他宽度和真实联合迁移尚未完成，不新增竞争系数。

### 2026-09-13 Lab：独立需求曲线接口

`narrow_stage_demand/request_curve.py`独立消费1/2/4T两场请求bank，固定full stripes、
H4096/F512及1/32copies，保留DRAM MiB/call与L2 events/call。默认只允许M12/48/96；
显式piecewise假设为相邻节点a,b之间Q(M)=Q(a)+(M-a)/(b-a)*(Q(b)-Q(a))。
显式late_linear将48..96斜率延伸到1718，仅为待验证候选；M<12拒绝，未借用其它宽度。
两场参数分别查询，不插值复用状态，不裁剪有符号测量，不声称插值包含尾块机制。
给定独立阶段耗时D，平均请求率为Q/D；D变大时平均率降低、总预算守恒。
此为whole-stage预算接口，不能解释为已知panel时间分布，未自动送入资源容量约束。
实测节点也未通过真实route迁移；插值/外推返回显式未验证标记。生产/default不变。
19项L1测试及72原始观测精确复现通过；120个组合数值检查不是L3泛化验收。

### 2026-09-13 Lab：实测时间线条件响应诊断

`narrow_stage_demand/fit_trace_response.py`仅在core_budget_joint的1/2/4T背景上，按实测
阶段重叠比例计算四项特征：共享DRAM读、同域L2 refill、共享/同域gather逻辑工作率。
阶段请求率假设Q/D_bg，D_bg为实测背景阶段耗时；乘固定前台T0后，以非负四参数拟合
第一场配对减速delta。前台窗口同样来自实测，故这是条件诊断，有目标时间信息，
不是可直接部署的自主预测。gather逻辑量不是实测DRAM量，不含独立写压力项。
需求1/32copies与场次1/2为四套独立情景；主要探索情景预定32copies/第一需求场，
不按分数挑选情景。所有未测M显式使用待验证late_linear，不改变独立T0。

第二场六条件MAE：W13 10.108us、W2 5.211us；同条件旧模型16.089/9.663us。
两者信息输入不同，此差值不是模型泛化改进证据。按背景宽度留出时W2 MAE为
10.654/4.696/10.839us（留出1/2/4T），不稳定。W13主拟合DRAM系数零、gather占主导；
不可解释为物理DRAM无效。四参数只有六条件，留宽度仅四训练条件；归一化条件数
W2留4T达353.15，参数辨识不足。未通过响应门槛，不接入planner默认。

### 2026-09-13 Lab：独立gather总响应约束

`narrow_stage_demand/constrained_response.py`在四项条件特征回归中显式约束
kg_shared+kg_local=k_independent，且所有系数非负。同LLC域的独立实验只能约束
此总和，不能自行分离shared/local。消去kg_local后回归矩阵为
[X_DRAM,X_L2,X_gshared-X_glocal]，目标y-k_independent*X_glocal，
约束0<=kg_shared<=k_independent。枚举非负最小二乘解和上界固定解取最小残差。
固定总量必须由调用方提供，不从联合y重新推断；保留有符号响应y。
返回降维矩阵rank/condition；秩亏不宣称物理参数已辨识。
8项解析边界/不可辨识/输入拒绝测试通过，尚未使用正式实验生成参数，默认不变。

### 2026-09-13 Lab：第一场独立gather响应冻结与约束诊断

独立4T/M48探针第一PMU按stage/reuse分组，使用两种gather M的条件中位，拟合
k=max(0,sum(x*y)/sum(x*x))，x=T0_gemm*Qlogical/cycle_gather，y为配对服务delta。
保留原始负y及无约束k，未读取第二PMU。k W13 1/32copy=.048226/.201719，
W2=0/.253004；零边界不证明真实响应为零。有限压力段、实测gather周期和合成路径
限制外推，不能直接映射真实4copy复用状态。
固定此总量约束旧联合诊断，主32copy假设第二旧会话MAE W13/W2=10.515/4.713us，
最大18.507/16.266us，仍不通过。W2同域4T背景高估16.266us；留4T训练外的MAE
15.732us，max29.114us。未改善宽度迁移，默认不采用；不重调已冻结独立系数。

### 2026-09-13 Lab：请求总量守恒的前缀时序对照

request_timing.py以Q(12)、Q(48)、Q(96)、Q(M)的差作为分段预算，时间边界
假设t(m)=t_start+(t_end-t_start)*m/M；整段预算守恒。这是均匀行进度假设，
不是实测panel历史；负read/refill增量拒绝而不裁剪。fit_trace_response的--timing prefix
仅运行预定32copy/需求场1情景，不改默认uniform路径。gather特征、y与T0保持一致。
固定原约束响应系数时第二旧会话MAE W13/W2=46.476/21.921us，劣于uniform。
分开允许重拟合响应后为10.251/4.543us，但W2 local4仍高估16.238us，未解决问题。
7项分配/响应测试与Ruff通过，仅诊断，不采用；下一步应直接检验GEMM竞争响应。

### 2026-09-13 Lab：单背景需求响应宽度留出候选

`gemm_pair_response/fit_response.py`只用背景1/2T第一PMU拟合，前台固定4T/M48，
W13/W2与前台1/32copies各有两系数，delta=T0*(kD*Rbg_solo+kL*L2bg_solo)。
R为idle-subtracted MiB/us，L2为events/us；特征来自独立后台与前台solo，不使用
joint阶段窗口/请求率。响应负y保留，系数非负，零压力回到T0。
4T背景排除于拟合，须用其独立solo标定输入评估；不读取其joint值校准。
第二场重复评分保留第一场的solo特征，不以第二场输入重校准。1/2T第二场MAE
.265138/.270108us，max.602187/.853621us，仅同条件重复。32copy前台两个stage
L2系数均零，不代表高压力或多team局部服务无效。4T留出、多team、真实route未通过。

### 2026-09-13 Lab：单背景4T留出两场完成

冻结1/2T拟合、以4T独立solo标定输入，4T两场delta MAE .434134/.377835us，
max .898440/1.122620us；零预测MAE1.376875/1.457500us。全部引用质量通过，
达到预定单背景5us/12us诊断门槛，保留此M48稳态候选，参数未回调。
`gemm_pair_response/decision.json`明确multiteam_transfer_passed=false、
production_adopted=false。独立系数直接多team迁移仍失败，需固定stage/M增加压力
检验非线性响应；不得将小范围宽度通过升级为完整planner模型已完成。

### 2026-09-13 Lab：多team需求聚合与响应分层诊断

`gemm_multiteam_response/diagnose.py`固定单背景系数，比较三种后台压力：
P_sum=n*R1；P_feedback=R1*lambda_group/lambda1；P_group为后台组独立实测率。
前两者区分是否考虑后台吞吐反馈，后两者差异检查每调用预算能否聚合。
同域L2项保留，跨域置零；DRAM共享。FG T0独立测得，不从joint y拟合。
P_group与lambda_group为条件诊断输入，不能充当任意新plan已知量。count4留出规则
保持；2项聚合单位测试通过，尚未生成正式结果或增加响应系数。

### 2026-09-13 Lab：饱和响应候选及否决

在第一PMU的count1/2/8上试验delta=T0*(a_s*R/(1+R/Rc)+
b_s*L/(1+L/Lc)*I_local)，R为GB/s，L为refill events/us，两个饱和尺度跨前台stage共享。
两stage各两非负敏感度，R/L尺度分别31个对数节点[1,512]/[10,10000]；零压力返回T0。
只拟合开发counts，不读取count4响应或第二场。冻结Rc34.296751GB/s、Lc2511.886432
 events/us，均为经验尺度，不作硬件容量解释。
后台组独立实测压力输入时count4两场MAE2.406925/2.721925us，max5.297006/6.037006。
改用单team独立压力相加、不使用组profile时MAE3.792611/4.092621us，零预测却为
2.15125/1.84625us。虽满足原5us/12us绝对门槛，仍不采用；单调饱和不能解释所有
非单调服务中位数。saturating_decision.json记录not_adopted，不按已见留出调参。
2项极限/位置测试及Ruff通过；尚无自主动时序或planner采用。

### 2026-09-13 Lab：动态事件模拟的独立refill通道

`dynamic_resource_candidate/simulate.py`复用已冻结DAG/gather-worker/阶段切换/发布状态机，
额外引入segment.l2_refill_events、refill_sensitivity和gather_l2_refill_events。
每个活动片段率r_i=Q_i/T_i0*v_i，局部压力按LLC域求和且排除同expert自身请求；
v_i的分母增加refill_sensitivity_i*k_refill_to_kind*P_local，所有参与者同时更新反馈。
字节预算与event预算分别积分守恒，事件率不换算为LLC字节。片段结束立即重算活跃集。
可选local_refill_capacity_events_us按域独立约束比例服务；低于孤立请求率时拒绝，
不偷偷改变新通道的无竞争基线。这是待校准比例策略，不声称硬件公平性或重叠机理。
旧shared-cap逻辑保留为对照，不将旧logical字节误称实测DRAM。

9项测试覆盖解析fixed point、同/跨域、相同总预算不同相位顺序、阶段起止、单expert
自身排除、域容量及非法预算；9passed、Ruff通过。关闭新通道时，一份既有33job
结构的事件、segment与字节交付逐字段复现旧引擎。尚未接入真实profile和planner评分，
参数准确度/完整计划泛化未验证，生产/default不变。

### 2026-09-13 Lab：独立需求接入与自主队列回放

`dynamic_resource_candidate/adapter.py`仅对T1/2/4、M12..1718适配独立read/refill预算。
前缀时间用冻结isolated模型的T(node)差；node为12/48/96及最终M。读/refill总量守恒，
原first/later响应敏感度、gather逻辑量和T0不变；首段边界不一致则拒绝。
写请求原值保留但本read-only消融不参与预测，未裁剪负测量。uniform与prefix显式选择，
不能称逐panel真实需求已知。refill通道供给已记账，当前仍用原响应系数（新增refill响应零）。

`replay_narrow.py`对8个窄team队列条件、31独立arrival场景自行推进阶段，不以联合
trace时间作预测输入；trace只在评分使用。原路径精确复现。第二会话阶段delta MAE：
legacy W13/W2=16.089/9.663us，uniform17.063/14.116，prefix16.128/8.764。
任务组完成MAPE .5854%/.5768%/.5693%，改善很小，不能作为全MoE或排序改进证明。
8/16T、小M<12及完整计划剩余任务尚未接入，不能称完整新模型完成。
最终adapter与engine共15测试通过，240次实际输入适配的边界守恒核验通过。

### 2026-09-13 Lab：完整计划需求消融与响应覆盖审查

后续adapter显式支持T8/16独立bank；8T节点12/48/1205，16T节点12/48/1718。
插值与外推标注来源，M<12默认拒绝；仅显式m12_anchor策略借用M12请求预算，
保留该M自身T0。该小M需求假设未验证。48个宽team测量观测及896次job适配检查通过，
engine/adapter/wide-bank共18测试通过；不代表真实route需求迁移通过。

replay_full.py对两个既有224-expert计划、31独立arrival场景自主模拟，固定T0、
入口、交接和旧响应。测量仅在预测写出后评分。两场历史数据的四项完成误差汇总：
legacy/uniform/prefix MAE253.366/143.102/397.204us，MAPE .877/.492/1.370%。
三版均预测anchor较快；预测计划差754.066/757.225/774.294us，实测953.080/1133.780us。
只有两计划，不构成新鲜泛化、排序改善或等预算搜索结论，不据此采用uniform。

structural_audit.json发现按任务T0累加约80%的W13/W2服务仍为零shared sensitivity，
全部refill sensitivity为零；局部旧响应可能仍存在。这是响应覆盖限制，并非80%的
关键路径时间无竞争或对残差的因果归因。后续必须独立验证后段响应及请求时序。
写请求仍省略、gather仍为logical预算、32copy独立需求与真实路径状态差异保留。

### 2026-09-13 Lab：容量与显式敏感度的交互审查

共享容量分配独立于显式sensitivity：当offered=sum(q_i*v_i)>C时，对q_i>0片段
再乘C/offered。因此零shared/local sensitivity仍可能减速，不应自动增加接收系数。
full_replay/capacity_channel_audit.json固定四臂首个独立arrival，仅移除旧容量参数，
完成时间降低1192.999..1678.991us；prefix与uniform的差由348.063/370.361us降到
-4.084/-3.894us。各零shared/local片段移除容量后服务回到T0（数值残差<1e-9us）。
这是模型结构消融，非物理瓶颈归因、精度评分或新鲜验收。旧C=202.01615989208221
KiB/us作用于混合的DRAM读与gather logical预算，物理口径未闭合；优先验证资源口径、
容量和剩余接收响应的分工，避免重复计费。默认不变。

### 2026-09-13 Lab：独立DRAM容量预算通道

为避免旧logical响应特征隐式充当物理需求，simulate.py新增可选dram_capacity_kib_us。
各GEMM片段显式提供dram_read_kib/dram_write_kib，gather提供对应逐worker数组。
任意一个DRAM预算出现或启用容量时，所有任务/阶段/worker的读写预算必须完整、有限且
非负；未提供不等于物理零。旧shared_capacity与新dram_capacity禁止同时启用。

定义d_i=(Q_i,read+Q_i,write)/T_i0。响应固定点先产生v_i，再对sum(d_i*v_i)>C_D时
的d_i>0参与者按C_D/sum(d_i*v_i)比例缩速，并随事件推进重算。旧read/write/logical
特征仍独立用于旧响应公式，不计入新容量；DRAM读写各自积分守恒。读写合用标量容量
及比例分配仍是待验证假设，不代表控制器读写服务等价或硬件公平规则。
容量低于任一孤立片段或同team所有gather worker同时发出率时拒绝，以保留T0。
新通道不自动改变响应系数，也不解决剩余响应与容量重复计费的标定问题。

30项测试通过，包括12项新DRAM检查。两完整224-expert计划首入口场景的旧事件、
片段、字节和refill逐字段一致，dram_legacy_parity.json记录。未接真实计划物理预算，
不能据此声称新预测准确度。独立16T gather预算20个状态/会话观测保存在
gather_dram_observations.json：同round idle差除native调用率，保留31个有符号样本；
未测worker归属、时间分布与真实route迁移，不自动填入模型。生产/default不变。

### 2026-09-13 Lab：平均容量可行区间反例

audit_average_capacity.py限定检验delta=T0*max(0,Roffered/C-1)，Roffered是独立FG/BG
各自扣同round idle后的cycle平均读写率之和，T0为FG平均worker服务的跨round中位数。
它不等同动态模拟器，也未考虑请求突发、cycle固定余项或计算访存重叠。
描述性容差epsilon=2us时，每条件要求C>=R/(1+(delta+epsilon)/T0)，若delta>epsilon
另要求C<=R/(1+(delta-epsilon)/T0)。delta+epsilon<0则正减速模型无解。

既有gather_demand_states两场各18联合条件：第一场全条件C下界168.379GB/s、上界
118.869GB/s；第二场168.471/119.426，区间均空。独立solo已达到的中位读写率为
134.814/134.725GB/s，仅是经验诊断下界，不是统计置信界或硬件peak。36条件有7个
区间与该下界不相容，其中一项贴边，不能脱离噪声强调计数。保留原始逐条件结果，
不按这些结果修改容量、T0或容差，也不据此否定动态通道；下一步需辨识阶段内需求
分布与可受资源影响的服务。3项解析区间测试通过，默认不变。

### 2026-09-13 Lab：独立基线与DRAM服务进度

simulate.py以可选segment.dram_service_us及job.gather_dram_service_us逐worker数组
启用双进度模式。所有阶段必须显式给出窗口L：正DRAM预算时0<L<=T0，零预算时L=0。
基线工作剩余c以单位速度推进，DRAM服务剩余d初值L、按容量分配速度v推进；
q_read/write=Q_read/write/L*v，仅d>0时发出。阶段在c和d均完成后结束，依赖/CPU释放
仍由原DAG状态机控制；任一请求完成或新阶段启动均重算压力。T0是冻结服务基准，
不宣称已测得纯计算成本；L是待标定有效窗口，不由历史失败kr/kw自动换算。

该候选能表达T=max(T0,L/v)的常压极限，但动态时使用事件积分而非预先固定v。
孤立请求率检查使用Q/L而非Q/T0，并对同team gather所有worker求和。DRAM请求完成
后即停止占容量，计算仍可继续；反之计算完成后必须等待请求才可发布后继。
为避免语义混用，本模式拒绝旧shared容量、refill容量和任意非零旧响应系数；
旧logical/refill仅按基线进度记账、无资源响应作用。纯DRAM比例分配仍待硬件验证，
尚不表示任意指令依赖/预取延迟/读写调度或真实phase window已辨识。

12项新测试加33项既有测试共45passed，覆盖隐藏延迟、请求完成后的压力释放、
计算先完时依赖等待、gather与窗口边界/缺失、孤立基线及混用拒绝。
旧steady gather_overlap的M180留出失败保持，不移植其参数；新结构未做精度采用。

### 2026-09-13 Lab：请求窗口辨识的实验条件选择

design_window_probe.py仅用gather_demand_states第一场IO32/32独立成本与请求预算，
按logical read/write份额分配到16worker（不是实测物理归属），考察L_i=lambda*T_i0。
给定诊断参考C=134.814005GB/s，为保持孤立T0，lambda下界为sum_i(Q_i/T_i0)/C；
M96/180/1718分别.212983/.625719/.737969，上界为1。C仅来自已达到solo吞吐，
不是标定硬件上限；这些范围是条件假设，不能作为测得的窗口置信区间。

两角色各short/full、4个M组合、8个偏移，共128次事件模拟。M180前台/M1718后台
同时启动的四假设减速3.008..23.747us；偏移350us时短后台窗口约零、长后台窗口
仍为3.008..13.167us；偏移500us均约零。选择0/350/500us分别检查受压响应、请求
结束边界和无重叠控制。M96/M96同步四假设仅差.672us，不优先采集。
该脚本只评价gather，W13/W2为终止占位，不能叫MoE完整计划。原生单次、可验证
偏移的测量实现和实际结果仍待完成；不从这些模拟更新模型参数或声明准确度。

### 2026-09-13 Lab：原生非对称偏移对照两场结果

gather_offset_probe实现单次FG M180/16T与BG M1718/16T，独立32copy buffer，固定FG64call
再BG64call准备；所有worker ready后共同epoch+2ms，偏移0/350/500us。线程回收延后到
所有计时调用完成。两场各5warmup31round×7条件结束，FG同round同偏移n0对照的服务
增量为7.626/9.270/-.023us和7.819/9.655/.159us。350us时stage包络仍完全重叠，500us
不重叠；这不是直接的DRAM请求观测。两场FG配对时序均合格，第二场BG-only有一个
预热及一个正式start跨度超界，完整保留并单独标记后台参考质量。

同round比较delta350-delta0，中位数1.823/1.656us、MAD .870/.671us。晚期响应没有
消失，并略强于起始响应；先前short后台窗口情景在350us约零的预测与本批不一致。
不过本批准备/单次状态与旧steady profile不同，不可按此直接判定真实L、唯一DRAM
原因或拟合采用long参数。原生worker窗口/空载控制为后续状态及资源响应辨识提供约束。
默认未改，完整计划与等预算搜索仍待完成。

### 2026-09-13 Lab：同域／跨域原生偏移辨识启动

gather_offset_locality固定FG16T/M180 CPU288..303，仅将BG16T/M1718置于304..319或
256..271，同NUMA3。两placement各自匹配准备/offset的无后台控制，主指标为350us
同round的(delta_local-delta_cross)；0/500us、BG耗时及实际stage重叠同时记录。
不同核位置也可能改变后台需求与状态，不把形状相同当作物理请求相同。
首烟测BG-only时序失败保留；原子状态分cacheline的v2重新构建，14cell烟测通过。
两场5warmup31round正式采集已启动，尚无结果、拟合参数或默认模型变更。

### 2026-09-13 Lab：同域／跨域偏移两场完成

gather_offset_locality两场各504调用结束，350us local/cross配对增量分别6.359/5.396us
和8.539/4.752us，阶段包络基本完全重叠；500us两位置均无重叠、差值接近零。
同round local-minus-cross增量350us为1.047/2.963us、MAD2.142/2.432us；0us为
3.121/2.992us。局部额外响应幅度不稳定，不据此拟合固定LLC惩罚。
FG无后台成本local约42us、cross约39us，说明匹配准备/放置流程也影响基线状态。
已分开扣各自n0，但不据形状相同断言物理请求相同，不能把剩余差额唯一归因LLC。
第一场有一预热joint及一正式FGonly时序异常，均保留；第二场时序全部通过。
下一步必须使用匹配单次状态的独立读写预算约束窗口/容量，避免混合需求变化和服务
变化。无PMU数据、无新响应拟合或默认采用，完整目标未完成。

### 2026-09-13 Lab：匹配单次状态的独立DRAM计数启动

gather_offset_pmu复用原单次gather，只增加ARMED/GO/DONE/ACK：准备结束后才启用
计数器，固定epoch为ready+2ms；全部调用结束后停止计数，随后才回收线程/校验。
Python CLOCK_MONOTONIC边界验证每worker begin/end被所有计数器启用完毕与开始
停用之间完整包围，保留每event的count/enabled/running；单次无需steady calls/s估计。
各event预算为(count_active-count_idle*enabled_active/enabled_idle)*32bytes后求和，
不混用不同controller的计数窗口。负估计保留，写计数不等于全部dirty store同步落DRAM。

六个独立角色条件（FG180两offset、BG1718，各local/cross）在同进程同buffer内随机
交错PMU开/关；四个匹配idle只需PMU。两场16cell×5warmup31round启动，预定各条件
配对service偏差中位数绝对<=5%，running>=.99，时序门槛10/5us不变。烟测完成覆盖
检查，但单轮服务差仍有超界，不作为扰动已通过的结论。正式结果待完成，无参数采用。

### 2026-09-13 Lab：单次独立预算两场通过测量门槛

gather_offset_pmu两场完成，各6条件的PMU开关配对服务偏差中位数绝对最大.627/.390%，
全部通过5%门槛；quality.py同时要求两侧时序无异常和读写预算中位数非负，原样本保留。
同round FG local-minus-cross读预算差：offset0为939.433/814.889KiB，offset350为
921.973/861.132KiB，服务差约3.0..3.5us。写差较小且方向不稳定，不能用一个统一
read/write倍率概括。相同M/T不等于相同独立请求预算，位置响应拟合须输入匹配状态。

matched_profiles.json保留12个session/role/placement/offset显式profile、每个31轮，
每轮用无PMU worker成本配同round PMU预算；两场不平均、不插值、不分配物理请求到
worker，不将负原始样本截零。范围仅native_single_gather_fg64_bg64_ready2ms_io32，
没有真实MoE历史迁移、同步写回或窗口L标定声明。5项计数/资格测试和来源/372观测检查
通过，模型参数与默认不变，后续窗口/联合运行和完整搜索验证仍待完成。

### 2026-09-13 Lab：匹配预算的窗口参数诊断

fit_matched_windows.py固定matched_profiles的无PMU worker T0及独立读写预算，以逻辑
份额分配物理请求（仍是假设），设rho=sum_i(Q_i/T0_i)/C，L_i=T0_i[rho+theta(1-rho)]。
C为140/170/200GB/s诊断网格、前后台theta各取0/.25/.5/.75/1，共75组。rho>1拒绝，
不改变T0或裁剪需求；只用DRAM容量与双进度，不启旧响应系数。500us复用350us的FG
profile作无重叠控制，不能称500us需求已测。900次模拟先写出，再读取既有联合评分。

仅session1的offset0 local/cross两个点选择参数：C170、theta_F1、theta_B.75，训练
MAE .283us。350us预测第一场local/cross1.835/1.712us，对实测6.359/5.396；第二场
2.163/1.975，对8.539/4.752。四组训练MAE距最优<=1us的参数均保留，晚期预测仍小。
事后让第一场0/350都参与开发，网格最优第一场MAE1.795us但最大误差3.962us；该结果
不再是350us留出，也不作为参数采用。当前网格显示前后期折中，不证明所有连续窗口
模型不可能成立；还需区分时变需求和容量以下的服务延迟。
PMU solo与旧joint来自不同driver/process，转移未闭合；这是回顾诊断而非新鲜验收。
2项参数化约束测试通过，生产/default及正式系数不变。

### 2026-09-13 Lab：联合请求量与command占用代理辨识启动

已有单次solo计数按event独立扣idle，再计算sum(occupancy)/sum(read_commands)，
FG cross/local两场约51..52/54..55，BG约61..62，重复较稳；不转换为CPU延迟us。
gather_joint_pmu复用相同准备/epoch/计数握手，新增joint0/350us及其同进程无PMU
对照，两个placement全24cell。联合字节残差Qjoint-Qfgsolo-Qbgsolo无需吞吐缩放，
因为每活跃角色恰好执行一次；command代理对照采用独立corrected计数的加权总比。
联合计数/残差均是aggregate，不能指定给FG；所有有符号时间和流量差保留。
构建与单轮烟测通过覆盖/数值/CPU/时序检查，但服务偏差的正式5%门槛未评价。
两场5warmup31round已启动，不拟合参数或改变默认。该实验检验需求是否可加与服务
代理是否变化，不等价于精确队列延迟、LLC事件或完整计划验证。

### 2026-09-13 Lab：联合计数分布与同调用状态关联

gather_joint_pmu两场结束，8组通过既定计数/服务偏差/时序门槛。读残差有符号中位数
约-1.3%..+.35%，但逐轮绝对误差均值21.2..32.5%、P90 53.0..70.4%，不能据中心接近
零宣称固定solo预算可加。BGsolo最大间隙描述性分组：低组约14MiB、高读组23..25MiB，
同调用服务中位数差33..38us；分组不是硬件机制或可用于预测的已知状态标签。
按两个PMU读组服务中心中点区分control时间，14..17/31配对落在相反时间组，说明
按round配对的control成本与PMU预算不保证同一状态实现。control没有读计数，不能据
时间组直接断言其物理请求状态。

same_call_profiles.json保留同一次PMU的worker成本/请求/command代理，另列另一调用
的control成本，不将二者合成确定性状态或独立取中位数。旧profile和拟合保留。
全部记录的occupancy代理差中位数两场为3.4..5.5；事后近似可加子集仍为正，但这
不是过滤后验收、前台latency或微秒换算，不能自动提取响应系数。

### 2026-09-13 Lab：显式DRAM请求服务响应候选

新增可选dram_queue_us_per_kib=k，仅在separate DRAM clocks下允许非零；其他旧响应
及refill容量仍禁止混用。每次固定点求解使用实际请求率q_j*v_j，排除同expert所有
worker自身压力，令v_i=1/(1+k*sum_other(q_j*v_j))，再施加显式DRAM容量约束。
仅请求进度受到影响，基线工作仍独立推进；零请求或请求已结束不受此项影响。
它是有效服务响应假设，不是从aggregate occupancy换算的物理CPU延迟。
5项新增解析/边界测试加47项既有测试共52passed；k默认0，没有拟合或默认采用。
必须先处理状态/协方差与容量-响应辨识，再做完整计划泛化和等预算搜索验证。

### 2026-09-13 Lab：独立状态分布与动态传播对照

state_distribution.py仅使用第一场solo profile，枚举31x31 FG/BG独立组合，保留每role
内部read/write关联。预测先写出后评分两场joint；读量边际W1/实测均值约2.1..12.0%，
写量约.8..3.2%。按预测分布最大间隙作事后状态分组时，组内读量W1约.2..2.2%，
主要分布差异落在组权重。联合状态分组只用于描述，不作为预测输入；不能从31样本
的比例差异直接认定因果状态转换或排除抽样波动。W1不等于逐调用误差或计划指标。

state_time_replay.py固定原C170/theta_F1/theta_B.75、queue系数0，将第一场同调用
PMU成本与预算的31x31组合传播到4个gather条件，共3844个分布场景与4个中位数输入
对照，49.279s。第二场不更新profile；PMU/无PMU实测分开评分。无PMU FG350us均值
误差仍-3.77..-6.26us，故状态关联修正不能单独修复晚期低估。BG均值误差约-9.79..
+6.83us，而中位数误差可到-37.73/+30.54us，混合权重使中位数不稳定；不能以W1
或均值代替原完整计划/选择验收。role间独立性与逻辑worker分配仍是假设，参数未拟合，
默认未变。3项分布距离测试通过，下一步检验服务响应而非仅重新选择状态权重。

### 2026-09-13 Lab：状态条件服务校准与无条件复核

fit_state_service.py仅用第一场数据选择参数。独立FG读量三分层、BG最大间隙两组各取
真实调用代表及经验权重，保留同调用成本/需求。训练损失按第一场PMU joint计数的
低/高组分别计算FG/BG平均服务误差；实际joint组只用于训练/诊断评分，前向预测
使用独立输入组及其权重，不读取目标组。角色间独立与逻辑物理字节分配仍是假设。

C∈{170,200,230}、theta_F∈{.75,.875,1}、theta_B∈{.9,.95,.975,1}、k∈{0,.001,.002,.004}，
共144组。容量-only和非零k分别选代表性状态最优者，再用各4x31x31场景完整复核。
容量-only C170/theta_F.75/theta_B1，完整训练MAE3.957us；响应候选C170/theta_F1/
theta_B1/k.002，完整训练MAE1.608us。第二场状态条件诊断3.546→2.350us，非盲测。
三种C在最佳代表性θ/k处误差完全相同，不能据选中的170宣称容量已标定。

无条件预测仍固定第一场独立状态权重。对无PMU结果，两场FG/BG各条件的均值误差
MAE5.623/5.543→2.921/3.566us；中位数误差MAE16.521/13.989→13.112/11.683us，
后台中位数仍有明显问题。第二场FG均值误差约-.490..+2.077us，晚期低估改善；窗口
与响应均有重标定，不能将全部改善单独归因k。代表性状态固定窗口2x2对照另存。
该M类Lab候选不替换默认，需独立容量约束、未见条件及完整计划/等预算搜索验证。

### Lab独立读容量约束与标定边界（2026-09-13）

可选dram_read_capacity_kib_us=C_read要求sum_i(read_rate_i*v_i)<=C_read，
仅对仍有读请求的actor作比例约束，write-only actor不消耗该读cap；读写预算各自守恒。
可选total cap仍约束read+write，若同时声明则要求C_total>=C_read；旧logical cap不能混用。
独立速率超过声明容量时拒绝输入。此结构不证明物理读写互不干扰，mixed响应需另标定。

dram_capacity_probe两场M1/1T四copy、20/40/60/79team读主导测量全部通过既定仪器门槛。
79team W13为294.217/294.177GB/s，W2为292.534/292.559；60→79增幅约1.3–2.0%。
描述性平台通过，非通用硬件峰值。此前170/200/230总cap不能解释成通用物理容量；
旧候选留作历史对照，读平台不可直接赋给mixed总cap。当前不采用新默认数值。
56项Lab引擎测试通过，完整计划/未见条件/等预算搜索验收仍待完成。

容量消融已冻结theta_F=theta_B=1、queue=.002与第一场同调用输入；比较total170、
read292、read295、no-cap，保持L=T0不变。逐样本legacy等价检查后评分；当前运行中，
不能从首个local0四组相同推断所有动态条件。完整记录见Lab报告capacity_ablation段。

### Lab物理GEMM预算适配（2026-09-13）

physical_adapter.py保留T0与DAG，Q_read/write由raw_request_nodes显式构造：prefix为
相邻累计预算差，uniform为总预算乘阶段片段T0占比。检查非负、有限、读分配一致与守恒，
不截断负值。L=window_fraction*T0（零请求L=0），参数必须显式传入并标为未验证假设。
gather预算及旧response字段不变，需调用方补齐provider并选择兼容响应后才可模拟。
四组历史计划输入共896expert/3152片段转换检查通过，7项定向测试通过，默认不变。
这只完成GEMM输入转换，不能声称全计划物理模型已验证或采用。

容量消融最终15376场景完成：四arm所有条件逐样本相同，legacy差0，不能在此数据
辨识容量；不能推广为高并发无需容量。真实runner四copy仅为weights；gather input固定、
packed-A由resident scratch提供，FP32 route workspace与其不同，采集须按此生命周期设计。

真实route历史输入已提取14target：保留同scratch完整前序及target token集合。直接
前序地址交集可能为0而更早前序交集很大，故不能将previous_M视为完整输入状态。
集合交集仅作已访问地址特征，不等同cache驻留；其他lane/前次调用仍需动态历史处理。

同scratch实际route链gap实验两场完成：0/200/1000us忙等对mean-worker服务的配对
中位变化最大.383us，但team envelope可变2.23us。后续物理需求计数需独立验证到达
与服务，不能从服务稳定推出包络稳定或直接替换真实T0；详见gap_decision及报告。

实际前序PMU pilot两场表明M4/8T预算不可辨识（第二场read MAD115.62KiB，
write18/31负值），M73/16T读中心196.88/199.90KiB较稳定。仪器服务偏差小不等于
物理请求可辨识。history-to-start约2.4ms含人为2ms epoch，先缩短等待重验，不采用
或裁剪小M预算；此前1ms gap检查不能证明该2.4ms协议等价真实路径。

去掉额外2ms后两场PMU短epoch完成，history gap约400us，M4 read MAD64.31/24.32KiB
仍较大，不采用或裁剪。下一步完整计划gather需求敏感度；已核对两计划各788GEMM片段
L=T0下最大独立读率86.920GB/s，不超过295读参考，但此可行性不能证明需求预算准确。

完整计划nuisance敏感度首到达场景完成：小M gather从0到2倍仅改变约19–40us，
但全局queue=.002使nominal计划增加约4.7–4.8ms。gather逻辑/RFO参考非物理边界。
phase对照显示W13/W2从低估转明显高估而gather仍低估，不能统一迁移gather系数；
应分离接收响应和可隐藏时间。全31arrival主要对照exec46618运行中，非采用/泛化验收。

全31arrival的248预测已完成：capacity-only nominal低估两计划约3.1–4.2%，全局
queue=.002高估约12–13.3%。小M gather 0→2倍跨度中位30.34/19.12us，最大34.18/19.18us。
全部arm选anchor，无新选择收益；先分离接收响应和重叠，不按总时间拟合全局queue。

### Lab接收侧queue分离（2026-09-13）

k_receiver分别对应gather、W13、W2；缺省回退原global k，显式0允许覆盖。
v_i=1/(1+k_receiver(i)*sum_{j not own expert}(q_j*v_j))，其后保持原容量约束。
同一shared请求集合供所有receiver使用，不把来源阶段误当接收阶段；separate clocks
及有限非负验证不变。6项新测试，完整Lab71passed；四个完整计划结果global与明确
receiver参数逐字段等价。生产/default未改变。

固定.002逐receiver消融显示gather-only约+258us，W13-only约+3.3–3.5ms；W2-only
总时间接近却具有明显阶段误差抵消，不采用。进一步固定参数、L_gemm/T0=.35/.5/.75/1
得到非单调时间，缩短L同时压缩请求时序并增加峰值压力。L不是已测硬件窗口，粗大M
片段的请求提前完成也不是已验证的跨panel预取能力；需独立区分隐藏服务与时序后标定。

### Lab请求结束诊断（2026-09-13）

可选record_dram_timing记录虚拟memory clock结束；默认不输出新字段，不影响事件或预算。
72项Lab测试通过，6次完整预测剥离诊断字段后与已冻结结果完全相同。
L/T0=.35时两计划669/724个GEMM片段提前结束请求，大M后续片段约后60%时间无DRAM
请求、最大约8.5ms。L同时控制Q/L速率和请求活跃区间，不是单独的服务隐藏比例。
该时序尚未由硬件验证，不能凭总时间接近采用。下一步固定请求时序检验接收响应，
保持服务隐藏与请求集中两个假设可分别检验。

固定L=T0的接收强度标定仅用anchor session1的W13/W2阶段绝对误差，不使用
makespan训练。粗25点在零边界最优（20.3879us），因此在0..0.0002用同训练集
细化36点，不能据粗网格零点宣称无响应。回顾验证与训练分离，非新鲜泛化验收。

固定窗口细网格36训练点完成：k_W13=.000025、k_W2=.00005，阶段MAE19.9929us，
比粗零点20.3879us改善约1.94%，不作显著性/泛化声明。已冻结后启动两计划31arrival
回顾评分（exec83759），不根据总时间重新选值。新鲜计划和搜索验收尚未完成。

fine固定窗口回顾完成：anchor28094.345us、control28861.951us，误差仍约-1.9..-3.1%。
第二场8T小M W13低估而16T较大M高估，说明常量接收系数残差仍分组，不直接归因T0。
已准备uniformish新lane顺序615101/615102和anchor控制，234expert/10x8T；预测冻结中，
硬件尚未启动，不按该holdout调参；此批不等于混合宽度或完整搜索验收。

uniformish新顺序留出两场完成，candidate MAPE2.273%、P90abs3.209%，capacity
对照1.204%/2.058%；均选实测候选最快order615101，regret0、lane诊断通过。
仅本批新顺序检查通过，候选准确度较差且无选择收益，不采用或按holdout调参；
下一步冻结参数验证混合宽度，完整搜索仍待完成。

混合宽度留出已准备：每40核[4,2,1,1]x5或[16,8,8,4,2,1,1]，无竞争cost的
list scheduling分配234expert，完整PlanV2资源/依赖验证通过。source-conditioned
gather使用实际token列表。首arrival预测完成，全31arrival在冻结参数下进行；
不以此预测充当实测或等预算搜索结果。

混合宽度两场留出完成并失败：narrow低估约14.5%、wide低估约11.3–11.5%，
总MAPE9.448%、P90abs14.570%，lane诊断亦失败；regret0不能抵消时间误差。
主差额在1/2/4T GEMM，不能直接归因T0或特定硬件资源。先做匹配无竞争/前序对照，
保持冻结模型和失败记录，不加任意宽度惩罚，不宣称泛化或采用。

混合失败后的精确形状direct/prefix对照四场已完成并通过数值/隔离trace校验，
但回传连接超时，备用别名不可解析，尚无本地分解结果；不得重启已完成实验或
宣称原因已确认。分析器保持gather服务与到达分离，较大M直接对照仅准备未运行。

2026-09-14精确形状基线对照已收回：M14..22、1/2/4T的direct两场基线MAPE
0.703%/0.948%，保留同team前序的prefix为1.552%/1.267%。1T/M16 W13第二场
T0=1680.33us、direct隔离1660.02us、joint4007.75us；prefix隔离1664.10us，
联合减速仍大。证据支持在这些点优先修竞争响应，不支持统一抬高T0；不唯一归因
DRAM/LLC，也不外推大M。M99..240直接隔离两场已启动，尚未评分。冻结参数保持，
详见joint_cost_model_20260911.md及receiver_mixed_baseline_contrast/analysis.json。

2026-09-14大M补充已完成：8目标M99..240、双GEMM、两场direct隔离基线MAPE
0.634%/0.652%，各点误差均在±3%内。narrow1T/M111 W13联合增量中位2534/2510us，
2T/M168为1197/1208us；不能由当前T0偏差解释。保持T0冻结，后续以局部/跨LLC
受控多team对照区分局部压力与全局DRAM响应。原4T/M48单后台校准不覆盖此问题。
16T/M155 W13在联合计划快约42/45us，但direct改变了前序历史，不视作竞争加速。
详见receiver_mixed_upper_contrast/analysis.json；无新参数拟合或默认采用。

2026-09-14启动1T竞争响应诊断准备：receiver_locality_response以M16/M112、
W13/W2、4copy前台和4T/M48/W13、2/8team、1/32copy后台作36条件位置对照。
局部/跨LLC后台核集合相差40核；固定工作量不等于固定实际请求率。主观测为同round
窗口mean-service之差，保留负值和离散度，不把稳态重复权重对照当成真实route T0。
旧4T/M48单调饱和迁移失败记录保留；新实验暂不拟合或替换响应公式。

2026-09-14 1T位置对照第一场通过：M16/W13在8个4T/M48/W13、32copy后台
下的同域/跨域增量44.44/27.30us，远小于真实mixed_narrow约2300us。探针最多33
活跃核且后台固定阶段，不能将两环境视作同压力。先验证第二场，再补齐双域总体
压力覆盖/后台形状及请求量变化诊断，不直接从这批局部差异拟合修复系数。

2026-09-14第二场1T位置对照完成：M16/W13同域/跨域增量45.06/28.15us，与
第一场44.44/27.30us接近，均远小于真实联合减速；M112/W13同域93.1→104.9us，
不把所有条件称作精确稳定。双域20条件三角色控制已通过烟测/原始数据复核，
两场正式对照已启动，保留固定32后台核位置对照及64后台核总压力对照。非加性
诊断为T_both-T_local-T_cross+T_solo，非唯一硬件归因；未修改模型参数。

2026-09-14双域第一场完成：M16/W13在32同域后台核下增量46.0us，32核分域
34.58us，64核分域434.4us；M112/W13相应99.9/94.5/1396.5us。逐round非加性
T_both-T_local-T_cross+T_solo中位360.65/1245.8us，表明高总负载响应不可由
单域低负载直接相加预测。仅第一场、无PMU，不唯一归因DRAM容量或请求量变化；
第二场验证中。原参数与T0保持冻结，未据此采用新公式。

2026-09-14双域两场均完成：64后台核的M16/W13增量434.4/436.32us，M112/W13
1396.5/1390.4us，支持高总负载非加性现象重复；无PMU不唯一归因容量或预算变化。
下一步仅准备receiver_dual_domain_pmu计数方案，覆盖全部65活跃核及同16个NUMA3
DDRC，加入BG-only/idle/同批无PMU对照；旧core范围漏8核须修正。未拟合或采用。

2026-09-14高负载PMU采集实现：65核L2 access/refill/writeback及16DDRC四事件，
共259事件，50个PMU/无PMU格点含BG-only/idle，500ms。三项本地测试及原始烟测
复核通过；最大单调用比例2.63%、前台仪器偏差0.757%。两场正式采集已启动，
尚无正式计数结论，不能由烟测或聚合DDRC推断前台DRAM预算；模型参数保持冻结。

### Lab intermediate-pressure validation (2026-09-14)

The dual-domain PMU experiment completed two matched 5/31 sessions with raw-data
parity and quality gates passed. High shared-load slowdown repeats without an
increase in foreground L2 refill/call; M112/W2 response shows cross-session drift.
This does not uniquely identify DRAM queueing or allocate global DDRC bytes.
The optional `receiver_pressure_curve` experiment adds 5+5, 6+6, 7+7 and
8+4/4+8 background-team placements with unchanged native binaries and 500 ms
windows. Fit placements are 4+4/6+6/8+8; 5+5/7+7/8+4/4+8 are held out.
T0, response coefficients, planner objective and production defaults remain frozen.
The new 80-cell protocol has four passing local checks; target smoke and raw
parity passed (maximum paired perturbation1.615%). Two formal sessions are running;
response holdout validation remains pending. This is not model adoption.

### Bounded response-shape diagnostic update (2026-09-14)

Both intermediate-pressure sessions completed and passed raw-data parity and
quality gates. Optional `receiver_pressure_curve/fit_response.py` compares
`delta = a * rho` and `delta = a * rho / (1-rho)`, with independently fixed
`rho = BG-only-read-GBps / 295`, restricted to `0 <= rho < 1`. Each nonnegative
coefficient is fitted only to session1 placements4+4/6+6/8+8, separately for
M16/M112 and W13/W2. Session1 measured pressure features are frozen for session2.
Rational heldout increment MAE14.483/18.676us versus linear165.852/161.749us;
M16 heldout underprediction remains systematic. These are descriptive-viewed
holdouts, fixed-stage measured-pressure diagnostics, not cross-M or full-plan
validation. The rational form is empirical and is not proof of physical queueing.
No production equation or T0 change. Receiver feature transfer and dynamic joint
validation remain necessary before adoption.

### Receiver-feature transfer limitation (2026-09-14)

A proportional transfer of the bounded rational-response coefficient from M16 to
M112 fails when scaled by isolated time or L2 refill/call. M112/M16 coefficient
ratios are3.209 (W13) and2.623 (W2), versus time ratios about6.9 and refill about5.3.
Solo DDRC read/call proxy yields M112 heldout increment MAE29.774/32.833us for
W13 but148.615/142.060us for W2. Thus an invariant sensitivity per byte/refill is
not established. These are retrospective diagnostics on viewed shapes; no change
to production equations. New M/history and request-exposure validation is required
before replacing per-shape sensitivity with independently derived features.

### Prospective receiver interpolation check (2026-09-14)

Optional `receiver_m_holdout/design.json` freezes `a(M,s)` as affine interpolation
between the existing M16/M112 diagnostic coefficients. Previous session1 BG-only
rates and295GB/s reference remain fixed. New M48/52 and96/100 validate full-block
versus four-row-tail sensitivity at6+6/8+8; M16/112 check concurrent anchor drift.
No fitting to these new M observations is permitted in the primary score. New
probe extends only the input whitelist; local grid/source tests and Arm
build/smoke/raw parity passed (max perturbation0.930%). Two formal sessions
are running with frozen predictions; measurements remain pending. This is a bounded empirical null hypothesis;
full-plan and dynamic-overlap validation remain outstanding.

### Matched block history/pressure interpolation diagnostic (2026-09-14)

Optional `tmp/joint_cost_model_20260911/receiver_tail_matched/fit_response.py`
implements the predeclared `fit_plan.json` conditional diagnostic. This does not
replace the active DRAM response, planner parameters, or isolated costs.
Only validated train1 (M52/M100, 6+6/8+8, 31 paired rounds) may fit parameters;
train2 is repeatability, M76 and7+7 are excluded from fitting.

For stage s, block history h and training pressure n, let
`d(s,h,n) = median_round(T_joint_block - T_solo_block)` in microseconds,
using PMU-off, block-timing-on samples. The difference is taken before the median.
At matching full-block histories, average these node medians across available
training M. Four-row tails retain separate h4 and h8 nodes. Independent BG-only
PMU-on idle-corrected read rates provide `p6,p8`, requiring `0 < p6 < p8`.
For each history, define `G(s,h,p)` by piecewise-linear interpolation through
`(0,0), (p6,d(s,h,6)), (p8,d(s,h,8))`.

The primary four-row-tail prediction for integer h in[4,8] is
`Delta_tail(s,h,p) = (1-lambda)*G(s,4,p) + lambda*G(s,8,p)`,
`lambda=(h-4)/4`. Full blocks support only measured histories h0..7.
Pressure is restricted to[0,p8]; unsupported rows/history and extrapolation are
rejected. All signed training increments are preserved without clipping;
a negative predicted increment is a diagnostic output, not a validated physical
speedup. T0 is unchanged and is not estimated by this function.

Controls: zero competition response and history-pooled first/reuse/tail4 groups,
with the same pressure interpolation. Validation cannot select the family or
change knots/coefficients. Conditioning on validation BG-only measured pressure
must be reported separately from autonomous pressure prediction. This scope does
not establish a queueing law, complete-plan accuracy, or planner selection benefit.
Both matched holdout sessions completed with full raw validation. All42 pressure7
nodes overestimate in both sessions; the linear-pressure candidate is not adopted.
A post-hoc q=p/(295-p) coordinate diagnostic reduces pressure error but retains
tail and history/context errors; it is not a new independent validation.

### Background-only composition selection (Lab, 2026-09-14)

`tmp/joint_cost_model_20260911/receiver_pressure_composition/selection.py` selects
conditions from independently validated BG-only screening, without foreground
response inputs. Grid: M12/48/96 x W13/W2 x symmetric5/6/7/8 four-thread teams
per LLC, copies32, two sessions,31 finite read/write samples per condition/session.
The caller must verify raw identities, counters, geometry and complete protocol;
the selector alone cannot authenticate measurement provenance. Missing/duplicate
cells or failed gates reject selection. Out-of-domain medians remain ineligible.

For each same-team-count pair with distinct(M,stage), define
`dr=max_session(abs(read_a-read_b)/max(read_a,read_b))` and
`dw=max_session(abs(write_a-write_b))`. Require dr<=0.02 and dw<=2GB/s in the
original calibrated[p6,p8] interval. Sort by(dr,dw,condition identities), greedily
choose up to two disjoint pairs. This is deterministic greedy selection, not a
maximum-cardinality matching objective. No qualifying pair means no matched-pair
claim. Signed idle-corrected writes are preserved.

For target pressures p6+f*(p8-p6), f=0.25/0.75, choose the eligible new condition
with smallest worst-session distance, requiring distance<=0.1*(p8-p6); break ties
by condition identity and deduplicate. Exclude original M48/W13 at6/7/8 from these
new-pressure targets. Freeze selected conditions before foreground acquisition.
The BG-only collector now implements all50 paired-PMU/idle cells per round with
two native background processes and no foreground process. The raw analyzer
checks the frozen protocol/candidate/native identities, exact counter-name set,
complete grid, native validation and finite service/counter values. It retains
signed idle-corrected rates and gates count resolution and per-role paired PMU
perturbation. Synthetic selector/protocol/raw-reduction tests:35 passed.
freeze_selection.py independently revalidates both raw JSONL inputs, requires
the predeclared distinct seeds and5/31 observations, and rejects failed gates.
Arm smoke exposed simplified synthetic DDRC names and a foreground-style native
service schema. Before formal data, protocol v2 uses actual
ddrc{0,2,3,5}_{0,1} names and the existing BG median_ns field; its paired
median-service perturbation gate remains5percent. Original failed smoke remains
preserved. Revised35 local tests pass.
It derives pressure bounds from the frozen candidate, preserves all48
condition/session observations, and writes an exclusive output containing input
and implementation identities plus deterministic selections. Reversing input
order leaves the output unchanged; no eligible condition is an explicit empty
selection, not a relaxed threshold. These hashes provide traceability, not proof
that a hardware measurement took place. Protocol-v2 Arm50-cell smoke passed numerical/CPU and quality checks
(maximum PMU perturbation1.197762percent; single-call fraction0.520833percent)
with exact local/remote raw-analysis parity. This validates this integration
sample, not formal repeatability or generalization. Foreground follow-up
integration, formal measurements, dynamic model and full planner acceptance
remain pending. No active equation or production default changed.


### Composition foreground follow-up protocol (Lab, 2026-09-14)

followup_protocol.py verifies a frozen selection by independently rerunning
freeze_selection.py on both original BG-only JSONL sessions. Mismatched output
or no eligible conditions rejects collection. New cells explicitly encode
(foreground M, foreground stage, background M, background stage, teams per LLC,
block timing, PMU). For K selected conditions, include:
4 foreground(M52/100,W13/W2) x(K+1, including solo) x4 timing/PMU controls,
plus(K+1) BG-only/idle x2 PMU controls, giving18(K+1) cells; K<=6 implies126.
Preserve foreground-first ARMED/GO ordering from the matched probe and disjoint
CPU319 foreground, local/cross4T groups ending316/276. T0 remains fixed.

measure_followup.py and analyze_followup.py retain separate FG mean_ns and BG
median_ns schemas. Validate complete grids, native/protocol/selection identities,
counter coverage/time, finite observations, and FG block-sum plus overhead
accounting. Gate FG/BG paired PMU and FG block-timing perturbation independently
at5percent, retaining the5percent single-call-resolution gate. No response is
fitted or evaluated by this raw analyzer. Synthetic full-grid, command/control,
selection-tampering and accounting/gate tests bring local total to50 passing.
Foreground hardware integration and frozen linear/queue comparison remain
pending; this is not model or planner acceptance.


### Frozen composition-response evaluator (Lab, 2026-09-14)

evaluate_followup.py revalidates foreground raw data and requires5/31 rounds
with seed616411 or616412. CLI selection is recomputed from original screening.
For each condition/M/stage/block, use PMU-off, block-timing-on same-round
joint-minus-solo increments; preserve31 signed samples, median and MAD.
Prediction conditions on median same-session BG-only read pressure, not
joint-run DRAM totals or observed foreground progress. Linear and q=p/(295-p)
comparators share frozen parent block/history values; neither fits capacity,
responses or T0. Original support is enforced before transforming coordinates.
Unsupported pressures retain all observed points and explicit supported counts
rather than disappearing from aggregate errors. Report stage/full-tail MAE/bias.

For each preselected composition pair, recheck contemporaneous BG-only read
medians (relative mismatch<=2percent), write medians (absolute mismatch<=2GB/s)
and original[p6,p8] domain. Report paired-round response differences, MAD and
observed-minus-predicted difference for both families, including failed matches.
A failed contemporaneous match is not matched-pressure evidence. One-session
differences/MAD do not establish significance or cross-session repeatability.
Endpoint/zero equivalence, curvature, raw integration, unsupported retention
and matched-pressure drift tests pass. Two-session conclusions, hardware
foreground integration and full-plan/planner acceptance remain pending.
run_formal.sh now sequences raw screening, frozen selection, actual-selected
foreground smoke, then two formal foreground sessions with all gates. It has
an exclusive output directory, no restart and a210min timeout.54 synthetic tests
pass on Arm; the new bounded formal batch is not launched pending resource
approval. Pipeline completion cannot substitute for model acceptance.
compare_sessions.py independently reevaluates two original foreground raw files,
requires both distinct predeclared seeds, and retains per-session group errors
and supported counts. A group can report lower queue MAE in both sessions only
with complete support in both. Composition residual sign agreement requires
contemporaneous pressure matches in both sessions and nonzero same-sign
residuals; it is descriptive, not a significance or mechanism test. Unsupported
conditions and disagreement remain visible. Seven focused tests passed, including
full raw revalidation and isolated sign/matching logic. The local formal runner
now writes comparison.json after both session evaluations; resource approval
remains pending and no formal run has started.


### Composition batch outcome and next contrast (2026-09-15)

Authorized batch completed all stages within approximately79min; five raw
analyses, frozen selection and cross-session comparison match local reanalysis.
No same-team composition pair qualified. The selectedM12W2/6+6 condition fails
the frozen queue candidate: every response error is negative in both sessions;
no stage/full-tail group improves MAE in both. No coefficient/default adoption.

Retrospective oldM48W13/7+7 vsnewM12W2/6+6 at similar BG read pressure suggests
additional configuration/history dependence, particularly W13 tails andM52 W2
tail. M100 W2 tail stays close. This comparison varies task count, date and
acquisition grid and cannot establish scalar-pressure insufficiency or hardware
cause. A separately prepared fixed54-cell interleaved two-configuration design
will test contemporaneous response; it retains same pressure-match thresholds
but deliberately permits different team counts and does not retroactively
change the original selection rules. Implementation and new measurement remain
pending, as do dynamic full-plan and equal-budget planner acceptance.


### Independent fixed-configuration implementation (Lab, 2026-09-15)

receiver_configuration_contrast snapshots the validated collector/analysis
structure without editing receiver_pressure_composition or its selection rules.
The new CLI requires --design, verifies the frozen design and binds
design_sha256 in raw metadata; fixed conditions areM12W2/6+6 andM48W13/7+7.
Two formal seeds616421/616422 are disjoint from prior data. Its54-cell grid
retains both FG stages/M52/100, solo, same-session BG-only and all timing/PMU
controls. Different team counts are deliberate, so conclusions concern whole
background configurations; no isolated task-count, M or stage attribution.
The new evaluator permits this unequal-team pair while retaining pressure
matching, support, finite-data and raw quality gates. Original same-team
selection is unchanged.

Four focused tests pass, including full two-session synthetic raw evaluation,
28 paired block comparisons, consistent pressure, second-session drift,
design tampering and rejection of old formal seeds. Old/new implementation
origin hashes are retained. The independent90min fail-fast runner has an
exclusive output directory and first gates a54-cell smoke. Remote synthetic integration now passes4 tests in3.48s and runner/design
hashes match locally; no formal directory exists. Actual native smoke remains
the first gated step of the prepared90min batch. New resource authorization,
measurements and dynamic/planner acceptance remain pending. No calibrated response, T0 or native/default behavior changed.

### Controlled background precondition protocol (Lab, 2026-09-15)

The completed fixed-configuration contrast passed both raw quality checks and
local JSON reanalysis parity. Queue-coordinate response improves W13 full/tail
and W2 full blocks, but not W2 tails; no adoption. Retrospective same-A data
show small solo changes and larger joint changes, with a repeated predecessor
association at M100/W2. Association is not cache-residency identification.

`receiver_precondition_control` holds native binaries, per-session allocations,
foreground M52/M100 and W13/W2 fixed. Before every target window, execute a
500ms background-only A=M12/W2/6+6, B=M48/W13/7+7, or idle precondition.
The target is A-joint or solo; repeat its precondition before each member of
an adjacent joint/solo pair. Reverse pair order on consecutive rounds and
between formal seeds616431/616432. Preserve all PMU/block-timing controls and
BG-only/idle targets:108 target plus108 conditioning windows per round.

For precondition c, use the same-round signed block difference
`Delta_c = T_joint,c - T_solo,c`; compare `median_round(Delta_c-Delta_A)`
separately per M, stage and block history in each session. Validate actual
sequence and conditioning native/window records before inherited per-context
quality gates. No coefficients, T0, active model or planner defaults change.
Foreground native lead-in remains >=64 calls; idle does not reset caches.
This tests conditioning under this protocol, not continuous expert history.
Local and target synthetic integration passed 12 tests. The approximately
194-minute batch completed within its 210-minute limit. Smoke and both formal
raw sessions passed all context quality gates; local independent reanalysis
exactly matches the three analyses and all 112 comparison records after JSON
normalization, with all 17 source/protocol hashes matching.

For tail-4 blocks, paired B-minus-A increments in sessions 1/2 (us) are:
M52/W13 -14.102/-14.955; M52/W2 4.805/13.809;
M100/W13 -17.264/-13.108; M100/W2 40.112/38.501.
This is descriptive conditional evidence, not an identified cache cost.
BG-only read pressure also changes with precondition, and session 2 has lower
read pressure but larger absolute tail increments under both A and B.
Thus controlled immediate background history does not remove session drift;
a universal history multiplier or pressure-only correction is not validated.
No fitted coefficients or model defaults change. See the experiment record
and receiver_precondition_control/formal/local_parity.json for evidence.
Complete-plan and equal-budget planner gates remain open.

### Fixed-schedule process repeat diagnostic (Lab, 2026-09-15)

The completed receiver_restart_control experiment held the six-round
schedule constant across four independent native-process sessions. Each session
has three warmup rounds and twelve formal rounds, repeating the six-round
schedule without restarting native processes between the two blocks. A/B
background-only preconditions, all four foreground M/stage combinations and
all PMU/timing controls remain; idle is not a conditioning state.

Compare paired positions between the two blocks within a session, and unpaired
node medians between sessions. Record native PIDs and reject reused process
identities, duplicate session identifiers or incomplete schedules. Time and
restart/allocation remain confounded across sessions; this diagnoses variability,
not causal cache residency. Original quality thresholds remain unchanged.
No cost-model coefficient or planner decision changes. Four short sessions
must not be substituted for unseen-plan and equal-budget planner acceptance.

All four sessions and the smoke passed the frozen acquisition gates. Local
raw reanalysis exactly matches remote JSON for every session and the 224
within-process / 336 between-session comparisons; source hashes match and all
12 native PIDs are distinct. See receiver_restart_control/formal/local_parity.json.
For B-conditioned M100/W13, summed block competition increments range from
906.640 to 1004.158 us across sessions, while paired within-session block changes
range from -2.633 to +6.132 us. BG-only read rates remain close (262.042–262.188
GB/s). This is not independently measured whole-GEMM wall time and does not
identify a restart or allocation cause. Tail-4 B-minus-A increments for M100/W2
remain positive (40.880–47.304 us); M52/W2 does not retain one sign. Do not fit a
universal history penalty or session label from these data. Next identification
must control allocation within process or validate an independent reference
calibration before continuous-trace and full-plan transfer. Model acceptance
and equal-budget planner gates remain open.

### Foreground allocation identity diagnostic (Lab, 2026-09-15)

receiver_allocation_control adds a standalone foreground probe with one or two
resident equal-content buffer banks. Bank selection occurs before a target
window; the timed kernel loop is unchanged. Each result records bank count,
bank identity, and all six virtual buffer addresses and sizes. Buffers are
4096-byte aligned, which differs from the original vector allocator and must
be checked using a one-bank control before attributing changes to switching.
Additional resident memory and physical placement remain possible confounders.
Virtual address identity is not a physical cache/DRAM mapping measurement.

Native validation covers all previously supported foreground shapes, both
stages, both timing modes, invalid bank commands, and switch-return address
stability/nonoverlap. The paired collector now shares two background processes among three resident
foreground actors (old, new single-bank, new dual-bank), retaining eight
precondition/variant contexts and bank-aware raw checks. Its 160 targets per
round each retain independent 500 ms A/B preconditioning, matched solo/joint,
and the original instrumentation gates. All actors remain resident, so the
extra memory footprint is a stated experimental condition. Two twelve-round
sessions with three warmups are bounded by a 180-minute runner. Real smoke
passed all eight-context gates with exact independent raw-analysis parity;
maximum PMU/timing perturbations were 0.601%/0.541%. Formal acquisition is now
started; no formal repeatability or model acceptance claim is available. This diagnostic does not add a model parameter or authorize
using bank/session labels as planner features.

### Full-load window table and window-aware quick cost (Lab, 2026-09-19)

Scope: Arm-codex NUMA3 80C, TP4 H=4096 F=512, SVE BF16 $\nu=16$, protocol
producer-hot A / cold B. Default dispatch is unchanged.

**Table.** For width $t\in\{2,4,8,16\}$ and uniform $M$, all lanes run the same
window at full load. The candidate policy `ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3` stores,
per route band and width, the chosen $(\omega_{13},\omega_2)$ and the measured ratio
$r(t,M)=T(M,t,\omega)/T(M,t,\text{full})$. V3 is measured with jemalloc preloaded and
purging disabled (`tmp/jemalloc_rerun_20260919`: the window grid, then a W2 sweep at
W13 = 1 tile, composed by the V2 procedure). $M\le16$ is DRAM bound and keeps the full
stripe. For $t=2,4$ a 1-tile W13 window wins at every $M$ from 24 to 720 ($r$ 0.61--0.83
for 2T, 0.73--0.97 for 4T); W2 windows of 4--8 tiles help at small $M$ (2T $-14\%$,
4T $-7$ to $-11\%$ vs W2 full at $M\le48$) and stay neutral to helpful at large $M$
(2T $-2$ to $-5\%$ at $M$ 288--720). $t=8$ windows $M$ 24--480 ($r$ 0.85--0.97);
$t=16$ only $M=24$ (0.92) and $M=48$ (0.97). The constant is not registered in
`_POLICIES`.

V1/V2 (`tmp/window_table_20260918`, `tmp/w2_window_20260919`) were measured under
glibc, where Plan V2's per-call FP32 route output page-faulted and was zeroed on every
call (v1.104); jemalloc times are 17--44% lower at $M\ge48$. That cost hid most
large-$M$ window gains (V2 kept W13-only or full at $M\ge236$ and 8T full at
$M\ge372$), produced the W2 large-$M$ penalty (up to $+7\%$) that is absent under
jemalloc, and caused the first-session $M=24$ bimodality. It also moved the full-stripe
best width to 8T; under jemalloc it is 16T for $M\ge48$, and with V3 windows the fastest
width is 2T at $M$ 12--48 and 720, 4T at 96--144, 8T at 192--480 (within 0.2--4%).

**PMU mechanism check** (glibc allocator; `tmp/window_pmu_20260919`, 2T/4T, $M\in\{24,96,288,720\}$). The full
stripe re-streams B from DRAM about once per 12-row M panel while the concurrent stripes
exceed the LLC: 2T reads 1.9/5.8/14/29 $\times B$ per call (panels 2/8/24/60). A 1-tile W13
window cuts DRAM reads by 27--75% and raises the L3C hit rate (2T $M=96$: 0.20 to 0.67);
L2 refill falls only 0--9%, so the gain is at the LLC/DRAM level, not L2. The cost is A13
rescans per window: at $M\ge288$ the windowed plan still reads 1.8--11 $\times(B+A_{13})$, and
DRAM savings turn into time sublinearly. The W2 window's large-$M$ loss is not explained by
A2 rescans (A2 fits L2); it coincides with 15--30% more DRAM writes. Total DRAM writes
(about 120--200 KiB per route) and an L2-refill-only bimodality at $M=24$ remain unexplained.

**Cost.** `IntervalPlanner(stage_window_policy=...)` is opt-in. With a policy carrying
`time_scales`, every quick/shared/LPT task cost becomes

$$
T^\ast(M,t)=T_{iso}(M,t)\,r(t,M),
$$

and lowering emits the same table windows. $r=1$ outside the table, for stage planners
and for policies without scales, so the Amazon V5 table and all defaults are unchanged.
The event model (`dag_makespan`, full search) remains window-blind.

**Validation (failed).** On six untouched routed layers (requests 008/016/022, layers
12/29; `tmp/window_validation_20260918`), every layer has one hot expert with $M$
1086--1984, so homogeneous quick stays at 16T/8T and the table changes it by at most
2%. With heavy-pinned heterogeneous shapes (hottest expert first on a 16T/32T lane,
the rest on 4T/8T/16T lanes), the fastest measured variant beats the current quick
choice by 2--15%, and windows contribute 0.5--5% of that (9% for `p16_r4` on r022 L12).
But the $T^\ast$ argmin passed the frozen gates only on 3/6 (G1: >=3% faster than the
baseline), 2/6 (G1w: windows attributable) and 1/6 (G2: within 3% of the fastest), and
was 4.6--6.8% slower on r008 L29. Predicted levels are 25--40% low and the per-layer
Spearman is 0.15--0.92: the isolated-LPT quick cost cannot rank pinned wide-lane plans.
Decision: no planner or default change; the table stays an opt-in candidate. Pinned
hot-expert shapes are the larger lever but need a cost model that ranks them.
Under jemalloc the same frozen plans and gates give G1 0/6 (the pick is 4--8% slower than
the baseline on the three layer-29 workloads), G1w 3/6 and G2 1/6; the fastest variant
beats the baseline by 2--10%, still pinned or windowed. The conclusion is unchanged.

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

- 2026-09-11：增加等宽team max-min服务分配对照，固定原参数冻结M4/M8前瞻验证；默认仍为比例分配。

- 2026-09-11：启动联合资源模型，新增真实路径逐样本基线/间隙审计与独立会话复测；冻结完整泛化验收目标，模型与planner仍在后续阶段。
- 2026-09-11：新增六条块级非线性竞争响应，冻结成本及首12锚点，补M8需求；
  新鲜留出增量改善但绝对完成仍回退，保留实验候选，不切换整体基线。

- 2026-09-10：冻结响应完成跨M与真实链式动态重叠验证；M8失准、M13/M48增量
  偏差相反，记录完成时间抵消现象，不重拟合、不切换planner基线。

- 2026-09-10：固定完整路径成本与需求表，只拟合8T/M12竞争响应；第二轮需求
  特征优于等预算任务数，保留混合局部回退及动态迁移未验证范围，不切换基线。

- 2026-09-10：新增8T分阶段/形状/复用状态的DRAM需求测量与区间表，记录路由
  对照未稳定复现、W2波动及独立需求不等于联合减速，不自动接入planner。

- 2026-09-10：完成8T单expert真实完整路径桥接，第二轮及同M跨输入验证；
  完整标定改善小幅基线误差，保留联合竞争待验证与M12波动，不切换基线。

- 2026-09-10：实验性拆分8T完整块/精确尾块基线与限定竞争增量，完成第二轮及
  固定真实trace验证；记录W13/M12迁移回退和竞争零命中，保持当前基线。

- 2026-09-10：完成实际1MiB/512KiB owner stripe的协同8T历史/尾块采集，
  记录行形状依赖和线程到达偏斜；尚未将数据回拟合模型。

- 2026-09-10：无team倍率新版显式加入16T gather延迟，记录阶段留出和固定计划
  改善及剩余排序错误；不改1T operator、不引入重复竞争放大。

- 2026-09-10：新版Lab移除所有宽窄team残差倍率，保留资源竞争；用固定计划
  揭示小M kernel低估、operator残差补偿与16T gather/间隙缺项。

- 2026-09-10：完成修正版M1/1T局部响应的同预算planner实验，高偏斜实测提速但
  共同计划排序仍错误，记录候选集变化与外推范围，不切换当前基线。

- 2026-09-10：完成冻结修正版历史3/6和比例12/26、26/12前瞻验证，记录较小但
  持续的比例方向偏差；模型参数及planner均未改动。

- 2026-09-10：先补纯竞争历史2/4，再固定纯背景拟合混合残差；记录第二轮留出、
  逐点回退与表格拟合的泛化限制，无竞争和planner基线不变。

- 2026-09-10：完成冻结混合修正的新比例/历史2、4前瞻验证；记录纯竞争插值与
  混合残差两类失败，未将验证集回用于拟合或切换planner。

- 2026-09-10：冻结无竞争/纯竞争曲线，新增有界19+19混合历史增量，记录第二轮
  回顾性验证与未测混合比例限制；planner基线不变。

- 2026-09-10：冻结六类无竞争曲线，仅重拟合M1条件竞争增量；记录36点回顾性
  留出、混合背景剩余低估和零竞争不变约束，未变更planner基线。

- 2026-09-10：实现六个参数化kernel历史函数及完整块/尾块组合；补测精确行数
  与完整panel历史，显式区分无背景验证和未验证的热状态压力交互。

- 2026-09-10：增加有代理下界和基线等式约束的有效部分重叠 Lab 原型；冻结
  奇数M/纯背景拟合，报告混合压力组合仍低估，不导出生产模型。

- 2026-09-10：完成M1–11/W13/W2统一可控竞争网格，固定核数对照M2/M120背景，
  保留两轮完整减速响应与负控制；不拟合计算/访存比例或修改模型。

- 2026-09-10：增加统一 mean-best 的新旧模型共同计划精度与所选计划实测对照，
  固定原评分池，复用相同 bridge 的证据并补测 high-skew 顺序差异。

- 2026-09-10：根据用户决策将 Lab full/strict CLI 默认设为新模型与 mean-best
  选择；保留显式 fallback 对照，生产选择器不变。

- 2026-09-09：增加冻结新模型评分池的 fallback/mean-best Lab 对照，逐项核对
  422个有序 DAG 与完整 ranking，保持生产选择器和模型不变。2026-09-10完成
  配对实测，high-skew改善22.93%、median改善12.55–16.18%，uniformish计划不变。

- 2026-09-09：增加新旧冻结模型的等预算 full/strict 搜索 Lab 对照方法；重新生成
  三条 route 的模板/顺序候选，核对实际 DAG 调用、冷搜索重复与关闭 early merge
  的硬件选择结果。生产 planner、候选空间、calibration 和 kernel 均不变。

- 2026-09-09：新增完整块/历史相关尾块 Lab 公式与回顾性验证；独立基础成本改善，
  真实联合 M13/M35仍有残差；生产模型不变。

- 2026-09-09：新增核数压力完整 MoE Lab 外推适配器，对46个历史方案样本进行
  不拟合回放，原模型预测零漂移。整体 MAE3.733→1.143ms，但 uniformish 退化，
  选型损失需单列；生产模型、校准与剪枝不变。

- 2026-09-07：按序固定workspace实验profile/身份隔离、同场确认anchor/elite、评估冻结v8
  误差、独立确认median smoothing。实验anchor更新为median2ab和high-skew189（61保留），
  uniformish不变；smoothing相对旧anchor约2.5%但不优于新elite，未采用。无公式拟合或新
  物理项，旧偏序仅诊断不授权硬剪枝；生产默认不变，容量增长仍低优先级。

- 2026-09-07：workspace方法补测46个旧VND/LNS代表plan共8场。greedy insertion出现稳定
  3.350/3.377%收益；median2ab43572仍是实测best且旧偏序标candidate_worse；high-skew
  LNS elite仍有效，uniformish仅对弱parent有效。绝对MAPE与增益误差需分开评估，不拟合。
  同时核实冻结backend_n_tile实际为16，旧workspace文档中的8为笔误，运行几何未变。

- 2026-09-07：8份历史order/交错相关frontier在workspace下完成16场无插桩回放。
  6个历史稳健正收益只保留median block一个，两个旧2% winner不再过gate。
  median smooth在一组过gate、同计划另一组未全过，待确认；38唯一计划对的冻结相对增益
  MAE诊断由5.128pp变3.591pp，仍有预测+12.041%而实测-3.282/-2.430%的反例。
  无重新拟合、候选生成、自动anchor替换或生产默认变化。

- 2026-09-07：固定workspace无插桩复测两条完整transfer frontier，各两场7计划，无通过原
  双场2%门槛的候选。high-skew跨域relocation由约-6%变持平，同域swap/relocation仍约
  -6.6%/-13%；保留anchor，分离两种输出生命周期证据，不重新拟合模型或变更剪枝。

- 2026-09-07：Lab固定max_tokens route workspace预分配并触页一次，独占lease、超容量拒绝；
  NaN覆盖正确性通过。两场steady-state收益anchor3.078/3.382%、transfer8.2–8.7%，
  初始化8.049/8.176ms单列。无模型、计划或生产默认变化；增长为低优先级TODO。

- 2026-09-07：输出预触诊断在两场中将M164 head W2从2.804/2.876ms降至0.539/0.542ms，
  预触之外minor faults中位数为1；完整清零自身9.1–9.3ms使inclusive时间回退17–22%。
  该异常优先归为输出first-write/page-state混杂，不用旧总residual拟合权重竞争项。
  不改公式、workspace生命周期或生产默认，保留Lab对照等待独立workspace验证。

- 2026-09-07：同宽transfer双trace各两场、每场7计划完成，输出一致且配对验证通过。
  median同域swap收益0.750/0.576%，未过2%；high-skew模型最佳代表实测回退5.8–13.2%。
  保留原anchor，不改公式或pruning；下一步仅对现有artifact复查isolated lane-load变化。

- 2026-09-07：Lab `transfer`按同域/跨域各采样swap和relocation，每类最多1024次尝试、
  24唯一候选；仅同宽、完整单域且非空lane，relocation不清空source。保留全部task元数据、
  CPU/width/window/merge，以新lane序列重建依赖。四类各保留model-best，剩余两槽按family
  顺序保留最大位置距离代表，最多6候选加anchor。记录完整state与affected-lane context。
  v8只提供评分，不拟合、不自动接受/剪枝；双session硬件gate尚待验证，生产默认不变。

- 2026-09-07：增加最新已测anchor输入校验与interleave/extended共享union对照，单策略
  cap不变，union显式cap13；生产默认与模型公式不变。

- 2026-09-07：分两阶段加入opt-in交错模板和更广顺序变换，保留旧模式及7计划预算；
  仅离线候选生成，无新硬件性能或生产采用结论。

- 2026-09-07：记录E60/10x8T合成M梯度和集中/交错实测验证；收益不随平均DDR压力单调，
  不改变cost-model公式或生产选择策略。

- 2026-09-07：新增GEMM平均密度均匀/非均匀直接对照，保持工作/时间和生产模型不变。

- 2026-09-06：新增isolated phase offered-rate二阶矩驱动的有预算顺序proposal实验；
  仅生成候选，包含增峰与大小交替对照，不作为cost-model修正或pruning依据。

- 2026-09-06：增加硬件winner周围的单层order-only Lab扩展，固定lane归属及72评分/7硬件
  计划预算；模型公式和生产neighborhood保持不变。

- 2026-09-06：新增独立离线双候选freeze/dedup/measure/winner流程，明确session一致性与
  actionable分离；复用已知三trace证据，不改生产默认或v8公式。

Context residual opening (2026-09-05): 独立 Lab candidate 增加冻结 v8 event baseline 上的
23-feature ridge residual、identity 校验、parent/group-disjoint replay 与范围外诊断。
不修改 physical calibration、Plan V2、production 或搜索评分/剪枝；首次实验与后续决定见
`optimizations/fused_moe_sve/results/context_aware_residual_experiment_20260905.md`。

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-08-18 | v1.28 | production analytical quick planner 增加版本化 $T_{iso}(M,t)$ 磁盘缓存，默认位于 `~/.fused_cpp/cache/moe_costs`。identity 绑定完整机器校准、解析模型版本、shape/TP/ISA/exact-M/width 域；命中只替代标量解析求值，最终 route plan 仍按当前输入重新生成。写入使用文件锁和原子替换，缺失、损坏、失配或权限错误均降级为原解析路径，不改变公式、候选、排序、Plan V2 或默认 BF16/W8A16 dispatch。 |
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
| 2026-08-16 | v1.25 | production planner 的 early-merge policy 固定为 `true`，删除 plan lowering 上逐 task completion-time DAG 与 routing-tail/burst gate；compute candidate、评分、剪枝、cache identity、Plan V2 三态 schema、native ABI 和手工 `true/false/null` runtime 控制均不变。m5 TP2、H4096/F1024/E256、2048-token TopK6 的 43 层真实 DSV4 路由上，旧 auto 在 43/43 层本来就解析为 on，on/off/auto 输出逐层完全一致，kernel 43 层总时间均约 1.24 s。双 rank 并发、每层 7 次 forced-miss 的 planner-overhead layer-median 从 23.565/23.549 ms 降至 1.424/1.401 ms；5 次同步全层 forced-miss planner+kernel 中位数从 2.973/2.968 s 降至 1.833/1.829 s，延迟降低 38.36%/38.38%。该 policy 是 m5 TP2 DSV4 的实测 operational default，不声称 merge service 已被 cost model 建模或跨 workload/机器最优。|
| 2026-08-17 | v1.26 | 新增 9.41 的 synthetic shared expert quick plan：standalone/TP 下把一个同形 shared expert 作为全 token、单位权重的内部 expert；有限搜索 `1x shared_width + N x routed_width`，要求 routed width 不超过 shared width，routed 总工作不少于 shared 时 shared 最多占半个 rank，并删除多 routed task 的全核串行 endpoint。shared 固定 lane 0 首任务且该 lane 完成后复用；distinct-route 解析 cost 由 native C++ mixed-width LPT 搜索消费。isolated cost、stage window、uncertainty 与 Plan V2 lowering 全部复用现有解析模型，cache identity 增加 shared mode；routed-only 候选和 Plan V2 schema 不变。M5 V4-Pro TP4 BF16 proxy 的 balanced/uniform/hotspot 21-run 中位数分别提升 16.31%/19.93%/75.88%，最大 cold/hit planner 为 23.574/1.120 ms，最大绝对输出差为 5.96e-8，因而通过 2%/1%/25 ms/3 ms gate 并启用。 |
| 2026-08-17 | v1.27 | SVE JIT W13 增加与标准 SiLU 分离 cache key 的 DeepSeek-V4 limit-10 clamped-SwiGLU epilogue；显式 `swiglu_limit=10.0` 才启用，0/None 保持旧指令流，static asm 和其他 limit 拒绝。该 variant 只增加逐 row-pair gate/up clamp，不改变 GEMM、packing、window、candidate、Plan V2 或解析资源公式；M5 H7168/F768 上 pure W13 的 M12/1T 与 M2040/16T 开销为 -0.00%/+0.12%，低于模型当前误差，因此首版复用同一 isolated cost。clamped V4-Pro TP4 balanced/uniform/hotspot 实测提升 16.99%/20.98%/78.59%，计划与 planner latency gate 不变。 |
| 2026-08-21 | v1.29 | 明确拆分 analytical quick/full：quick 继续以 homogeneous isolated-LPT 在有界时间内求较优解；full 恢复全部 strict mixed-width/temporal-order 及派生 dynamic-tail 的 phase-DAG 搜索，并把 quick 胜者作为额外 strict baseline。修正解析系统误差被错误按 `sqrt(waves*profile_runs)` 缩小的问题，同时把 analytical full 主目标固定为 minimum expected makespan，不再在大范围不确定区间重叠时按 active working set 改选。Arm-codex NUMA3 80C DSV4 上，one-head safe-full cold/E2E 为 `584.9/37.033 ms`，但不具备 full 语义；完整 full 搜索 `142` strict+`327` dynamic，cold 约 `42 s`。模型第一名 `(16,16,16,16,8,8)` 实测 `34.418 ms`，比 quick `37.141 ms` 快 `7.91%`；前六名实际最好 `(16,16,8,8,8,8,8,8)` 为 `33.907 ms`，模型第一名 shortlist regret `1.51%`。旧 uncertainty-overlap/working-set 选择会改选 `(32,32,8,8)`，实测 `40.725 ms`，因此不符合 full 的最优目标。Plan V2 ABI、production quick runtime 和数值语义不变。 |
| 2026-08-21 | v1.30 | 将既有 `NativeQuickPlanner` binding 拆为 `_C`/MoE-only `_moe_C` 共用的 module-local 注册，production 优先从 `_moe_C` 加载并保留 `_C`/Python fallback；候选、LPT tie-break 和 Plan V2 不变。新增显式 `MoePlannerRuntime.initialize_planner(max_routes)` 生成完整 dense `T_iso[M,T]`，并让 production runtime 默认关闭低命中率 route-plan cache。Arm-codex NUMA3 80C、captured DSV4、223 active/28 distinct M：`initialize_planner(2048)` 生成并落盘 16,384 个标量耗时 `3.452 s`；之后无 route cache 的 public `plan_for_dispatch()` 1000 次 median/P90 为 `4.584/4.612 ms`，其中内部 C++ quick 约 `1.20 ms`。直接 planner 对比中 Python quick 为 `7.25 ms`，C++ quick 为 `1.60 ms` 单次及 `1.056 ms` 1000 次 median；固定 8T greedy 为 `0.199 ms`。9-case operator-only 对比 fixed 8T：uniform/active8/active16 为 `-2.86/-15.54/-11.31%`，active32/active64 为 `+4.57/+0.65%`，active128/tiered/DSV4 的同计划差异为 `+0.17/+0.37/-0.13%`，bimodal 为 `+22.13%`。因此 C++ latency gate 通过，但 quick 质量不支配 fixed greedy；active8/16 的错误 `40T` 选择要求补宽 team isolated residual 后再宣称跨分布收益。 |
| 2026-08-21 | v1.31 | 新增 supported runtime control `FUSED_CPP_MOE_PLANNER_FIXED_THREADS`：正整数值强制 routed-expert production planner 只生成对应 homogeneous LPT greedy shape，未设置或 `0` 保持多宽度 C++ quick。`8` 在 Arm-codex 80C 上对应 `10x8T`，用于与当前 quick 做线上回退和 A/B；Plan V2、full offline、synthetic shared expert 和默认行为不变。captured DSV4、dense `T_iso` disk hit、无 route cache 的 public `plan_for_dispatch()` 1000 次 median/P90 为 `2.698/2.708 ms`，内部 fixed planner 约 `0.257 ms`，shape=`10x8T`、backend=`cpp_fixed_quick`。 |
| 2026-08-31 | v1.32 | 仅更新实验闭合状态，不改变公式、候选、剪枝、schema、ABI 或默认 runtime。AmazonC5192Cores NUMA1 的五进程独立 packed-B repeated-scan probe 得到 private-L2 retention knee `0.662x`、miss floor `17.37%`、2/4 MiB miss `68.03/79.80%`；替换 transferred prior 后 isolated MAPE、contention MAPE/P90、max shape regret 为 `7.96/10.85/16.08/9.21%`，相对旧 prior 仅改善 contention P90 `0.89` 点且 regret 不变。完整 high-skew TopK trace 上模型选择的 16T LPT 为 `22.857 ms`，候选集中已有的 8T reverse-even 为 `17.282 ms`，配对降低 `24.43%`，manual candidate 的预测/实测 rank Spearman 仅 `0.692`；因此当前主要缺口是 wide-team concurrent pressure 与 temporal-order ranking，而非 retention 或候选覆盖。三条 captured trace 的独立 W13/W2 two-stage 最优均为 `8T/4T`，但仍比 best whole-expert 慢 `1.93%--5.87%`，故保留为 conditional candidate/control。完整记录见 `optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md`。 |
| 2026-09-01 | v1.33 | analytical full 增加仅作用于未显式 shape 的离线搜索的 one-step width uncertainty gate：先求 expected-makespan winner；若其最大宽度大于 8T，则在与 winner 系统误差区间重叠的候选中把最大宽度最多降低一个已校准档，并在该档内仍最小化 expected makespan。该规则不使用 active working set 作主 tie-break，不改变 quick/request path、候选、剪枝、Plan V2、cache identity、kernel 或 empirical/native planner。正式 runner 固定 commit `b219627`、Arm-codex NUMA3 80C、三份完整 2048-token TopK6 trace、5 warmup/31 randomized paired runs/4 rotating weight copies；原 full 到 gate 为 high-skew `42.091->36.525 ms`（paired `+15.27%`）、median `40.643->34.433 ms`（`+17.95%`）、uniformish `33.859->31.987 ms`（`+5.69%`）。相对 measured candidate set 的 regret 为 `0/0.26/0%`，全部低于 5%，输出逐位一致；run id 为 `20260901T080803Z-arm_codex_internal-arm_high_skew_closure-b2196270211b`。 |
| 2026-09-01 | v1.34 | 在既有 cold-phase CP-SAT 上增加离线 LLC-domain shortlist master：只搜索 whole-expert width、single-domain placement 和 start/order，window 继续由确定性 policy 给出；当前 one-step full plan 的 width vector 先投影成 domain-feasible incumbent/hint。独立 60 s proof 输出完整 surrogate UB/LB/gap；10 s pool root 固定非热点 interval、保留 top-16 route expert repair，并以 start-vector forbidden assignment 生成 32 个 order/width/domain 异构 schedule。二维 contiguous-core lowering 不下沉 modeled release time，完整 analytical event model 重排后只实测前 8 名。Arm 80C 三 trace 均得到 32/32 lowering、proof gap `1.507%--2.345%`、最大 pool regret `0.330%`；high-skew/uniformish 相对 one-step 为 `-10.380%/-0.638%`，但 median 回退 `+2.388%`，未过 2% gate，故不扩展 43 层/第二 Arm。runner 同时输出 calibrated resource relaxation、fixed/legacy/one-step controls 和 `T_plan+T_execute`。该路径为 Lab/offline paper oracle，不改变 production planner、Plan V2、kernel、schema、ABI 或默认 dispatch；正式 committed runner 尚待代码 review/commit 后重跑。 |
| 2026-09-01 | v1.35 | 将 production quick planner 的原始 homogeneous LPT greedy strict DAG 作为 exact fixed branch 加入离线搜索域，与 domain-aware SAT strict 分支组成不混合的 plan-level union；greedy 原始 width、core placement、order 和 dependencies 不做 domain projection。两分支分别用 cold-phase CP-SAT 求 UB/LB 后取最小，因而在相同 surrogate 下严格不劣于 greedy。第一阶段只测 union proof 产生的同一个 60 s SAT incumbent 与一个 pure greedy fixed plan，不启用 tail pool、tail repartition、stealing 或 resize；one-step projection仅作为 domain search hint。Arm 80C 三条 2048-token TopK6 trace 的 union gap 为 `2.071%--2.129%`，全部选择 SAT；31-run paired SAT 相对 greedy 的 high/median/uniformish 中位收益为 `2.21%/12.49%/7.27%`，三条均 `31/31` 获胜。该结果只覆盖 fixed strict plan class；SAT 相对 one-step strict control 在 uniformish 仍回退 `3.89%`。动态尾池留作第二阶段在线 recourse，不改变 production 默认、Plan V2、kernel、schema 或 ABI。 |
| 2026-09-01 | v1.36 | analytical heavy event model 增加 placement-aware LLC path：planner 将 logical core interval 映射到真实 CPU IDs；event 按 owner-thread 比例拆分每个 task 的 LLC working set/demand，逐 domain 计算 capacity miss、service pressure 和 dilation，再施加 rank-level LLC fabric cap；task spill fraction 只影响自身 spillable DRAM，DRAM 仍为 NUMA-rank global，跨 domain gang 取最慢 domain dilation。旧 unplaced API、无-topology fallback、empirical/quick planner、Plan V2、runtime 和 kernel 保持不变。Arm 80C strict proof-plan 的 event 预测 SAT 相对 greedy 为 high/median/uniformish `-2.47/-2.98/-6.20%`，实测为 `-4.83/-10.92/+1.89%`；high/median 排序方向闭合，但 uniformish 仅 `4/31` 获胜并回退，证明剩余主要缺口是 wide-team concurrent pressure/width scaling，而非 LLC placement 丢失。placement-aware full planning 时间相对旧版约增加 `1.7--1.9x`，该实验未通过三 trace no-regression gate。 |
| 2026-09-02 | v1.37 | placement-aware heavy event model 增加离散 wide-team residual：以非 gate 层分别拟合 single-team internal dilation $B_t$ 与 full-cohort total dilation $S_t$，active event 使用 $B_t+(S_t-B_t)q$，未校准宽度不外推；4/8/16/32/40/80T 的 $S_t$ 为 `1.000/1.303/1.470/1.619/1.821/2.306`。CP fluid master 保持原 cold surrogate，其 gap 不冒充完整 event proof；lowering 固定 width/domain，先保持 fluid start 求 contiguous placement，失败时才最小延迟并标记 `*_DELAYED`。最终 event union 显式保留 full、one-step、exact greedy incumbents 与 CP top-8。commit `6b6b4d1` 的 Arm 80C 正式三 trace run 得到 proof gap `4.02/4.99/4.81%`、measured union regret `0/0/4.03%`，相对 one-step 和 greedy 均无回退且对 greedy 全部 `31/31` 获胜；但三条最终都保留 full incumbent，high-skew 中未选中的 CP plan 实测再快 `4.03%`，故只关闭 incumbent-plus-shortlist 的近似计划 gate，CP-only 与 temporal/lowering quality 仍 open。见 `optimizations/fused_moe_sve/results/arm_codex_80c_wide_team_strict_gate_20260902.md`。 |
| 2026-09-02 | v1.38 | 为event-guided VND/LNS增加内部version-1 canonical executable state，不改变搜索候选或选择结果。state将rank表示为gap-free contiguous fixed-width lanes及其whole-expert序列，独立记录ordered physical CPUs、contiguous LLC-domain partition、per-task windows和early-merge三态；lane依赖唯一派生，hash覆盖全部执行语义。现有跨LLC-domain lane原样保留并记录相交domain，不为了新抽象改写incumbent。适配器只接受strict/fixed/unsliced/lane-chain planner result，显式拒绝tail pool、route slice、resize和一般DAG；result→state→tasks/Plan V2逐字段往返并保持analytical event score。该步骤只建立后续邻域搜索的合法性边界，无性能或近似最优结论。 |
| 2026-09-02 | v1.39 | 增加Step-1 executable order-neighborhood审计，不改变production planner。五类move保持lane topology、width、window和early merge不变，并直接生成strict fixed Plan V2；canonical hash对跨算子重复候选去重。analytical model新增只读`explain_dag_placed`，与placed scorer共用原placement校验并暴露已有LLC-domain/team-pressure event。criticality采用tail-weighted event duration乘task phase dilation，仅控制等预算候选采样，不进入objective。正式Arm suite固定比较32个critical与32个uniform-random experts、每算子64个event评分、每组event top-4和31轮硬件paired measurement；在结果完成前不进入VND或ALNS。 |
| 2026-09-02 | v1.40 | Step-1 order-only neighborhood gate在commit `5252d67`、Arm-codex NUMA3 80C和三条2048-token TopK6 trace上完成。每条生成约9.2k--14.1k eligible proposals，每个critical/random arm实际event评分266--309个候选，吞吐5.64--8.60 plans/s；24个event-top实测候选没有一个paired P10为正。event/hardware Spearman在uniformish/median/high-skew为`0.143/-0.156/-0.690`；high-skew中event预测`0.068%--0.115%`收益的8个候选实测全部回退，最大`14.50%`。critical event-improving fraction为`22.6%/20.4%/48.8%`，random为`24.6%/43.3%/47.4%`，没有一致富集。故当前event model下拒绝进入order-only VND/LNS；先修复temporal-order ranking，或转向template-level global search。该结论不否定canonical executable state，也不改变production planner/runtime。 |
| 2026-09-03 | v1.41 | 为修复1T长lane的temporal critical-path crossing，analytical runtime overhead增加可选离散`by_width` override；旧profile与未命中width保持原global fixed/route overhead，persisted `T_iso` identity覆盖新字段。独立Arm 80C三态probe以同一任务/placement和31轮随机交错phase trace拟合1T `expert_fixed_ns=147999.019`、`route_ns=10471.803`：all-delayed/wide-only/all-concurrent victim实测`25.491/25.518/26.251 ms`，修正模型为`25.419/25.793/26.621 ms`，误差`-0.28/+1.08/+1.41%`。该参数尚需绑定commit的三条真实trace holdout；在通过前不恢复Step-2 VND。 |
| 2026-09-03 | v1.42 | commit `e74e187`的冻结v4 holdout显示width-specific overhead只部分修复point ranking：high-skew Spearman从`-0.690`升至`0.357`且旧critical relocation反例消失，但新random 1T swap/relocation仍实测回退`13.59%/3.50%`；median因full shape变为`1x16T+8x8T`后Spearman为`-0.756`，uniformish纯8T分数按设计不变。raw trace证明新反例仍在scheduled compute而非early merge。后续executable decision增加`max(event, (1+uncertainty)*max-lane isolated sum)` guard和2% minimum actionable gain；在已测24候选的离线重放中，high-skew Spearman提高到`0.548`并将13.59%反例判为`-1.016%` robust gain，三trace均无候选达到2%自动接受门槛。该gate待绑定commit正式重跑，不宣称point event排序已完全准确。 |
| 2026-09-03 | v1.43 | uncertainty-aware executable decision在commit `d51cb0e`正式复验：Arm 80C uniformish/median/high-skew均因best robust gain仅`0.666%/0.872%/0.153%`而保留baseline；相对同轮measured shortlist best的regret为`0.242%/0.578%/0.516%`，全部低于2%，24个候选仍无paired P10为正的稳定改进。high-skew point Spearman相对原v3从`-0.690`改善到`0.690`；median/uniformish为`-0.756/-0.048`，说明亚百分点point ordering仍不可识别，安全选择闭合不能冒充精确排序闭合。冻结v4在正式三态calibration rerun的all-delayed/wide-only/all-concurrent误差为`-2.33/+0.22/+0.61%`；当次重新回归参数有漂移但未用于holdout。Step-2 VND继续关闭，只有robust gain超过action margin的未来邻域才允许自动接受。 |
| 2026-09-03 | v1.44 | 增加Step-3 topology-preserving width-neighborhood审计框架，不改变production planner。只允许完整位于单一LLC domain内的等分lane split、相邻等宽lane merge和已校准相邻宽度lane间whole-expert migration；已有跨domain lane保持但不能作为width move端点。split/merge仅重排受影响lane，migration保留既有task相对顺序并枚举目标插入点；所有改宽task重新读取确定性stage-window policy。order-only、width-only、combined三种ablation共享canonical去重、placed event score、lane uncertainty guard、2% gate及同轮硬件shortlist。正式三trace结果完成前不进入width-VND/LNS，也不作性能收益结论。 |
| 2026-09-03 | v1.45 | Step-3 prefinal direct-sync Arm 80C三trace审计完成：uniformish/median的12/16个硬件shortlist均无paired P10为正的稳定候选；high-skew 13个shortlist中两个domain-local lane merge稳定提升`1.436%/1.248%` paired median、`0.262%/0.340%` P10，说明width邻域具有局部实测价值。但模型只预测两者`0.0197%/0.0130%`，反而把预测`+0.110%`、实测`-0.446%`的split排为width第一；median/high-skew combined Spearman为`-0.018/-0.011`。三trace无robust gain达到2%，故全部保留baseline，measured shortlist regret不超过`1.449%`。保留width operators作Lab候选，但在独立校准`1T+1T -> 2T` merge/concurrency transition并完成commit-bound复验前不进入width-VND。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_width_neighborhood_audit_prefinal_20260903.md`。 |
| 2026-09-03 | v1.46 | 独立建模并校准high-skew中的`2x1T concurrent -> 1x2T serial` transition。三组合成短任务在无背景与4x16T+14x1T满载背景下的实测dilation对pair/merge分别为近似相同的`1.09/1.09`、`1.07/1.09`、`1.14/1.14`，否定“缺少额外1T slowdown”，并定位为shared-resource model对2T background dilation高估。新增可低于1但保持final dilation不低于1的离散narrow-team correction；旧calibration和未校准width保持identity。`merge_isolated`拟合2T `expert_fixed_ns=88799.963`、`route_ns=5909.533`，完整event求根得到full-cohort correction `c1=0.784898`、`c2=0.532344`。三组pair/merge background holdout误差由`+5.43/+17.62%`、`+3.05/+17.55%`、`+13.23/+27.71%`降至`+0.12/-1.89%`、`-0.63/+1.01%`、`0.00/0.00%`；isolated pair保持`-2.85%--+2.37%`。模型名升到v6；冻结后的high-skew event-only预检查未再调参，真实三trace硬件仍为holdout，必须从同一commit正式重跑后再决定width-VND。 |
| 2026-09-03 | v1.47 | narrow-calibrated Step-3 suite从commit `1fcbc7b`完成正式Arm 80C三trace holdout。uniformish/median/high-skew均因best robust gain仅`0.666%/1.476%/0.278%`保留baseline，combined measured shortlist regret为`0.386%/0.637%/0.983%`，无selected regression。首轮唯一positive-P10候选是模型排名第一的high-skew `16T -> 8T+8T` split，paired median/P10为`+1.037%/+0.334%`；同commit独立repeat为`+1.271%/-0.632%`。repeat中一个merge为`+1.437%/+0.433%`，但首轮同plan为`+0.864%/-0.813%`，故没有候选跨两个formal session保持positive P10。uniformish/median/high-skew首轮event-hardware Spearman为`0.629/-0.003/0.049`，说明sub-2% width delta非平稳且point ranking未闭合。width邻域仅作为Lab搜索空间保留，不进入deterministic VND，也不降低2% action gate。主run id为`20260902T155347Z-arm_codex_internal_temporal_overhead-arm_width_neighborhood_audit-1fcbc7b130f0`，repeat为`20260902T161403Z-arm_codex_internal_temporal_overhead-arm_width_neighborhood_audit-1fcbc7b130f0`。 |
| 2026-09-02 | v1.48 | 增加不改变analytical mean公式的离线anchor-relative pairwise report：以candidate相对anchor的paired gain residual绝对分位数构造context→family→global回退区间，仅当完整区间越过anchor才声明better/worse，其余保持incomparable。same-v8前次session拟合、新session复验的41个pair中，3个满足硬件2%可分辨标准，但partial order对三者均不下判断，`0` false dominance且`41/41` incomparable；top-8仍保留三trace各自measured best，但当前关系安全而无判别力，禁止接入VND/LNS。配套two-level evaluator的exact/screen吞吐为`6.34--9.90`/`186--349` plans/s，预计减少`42--85%` exact calls并加速`1.57--5.06x`，但`8/16/32` budget均漏掉uniformish measured-best，故screen fidelity gate失败。独立1T lane-head swap probe在mixed `4x16T+14x1T`背景下的四组original-v8 residual为`-1.00/+3.76/-0.96/-0.25`个百分点，刻意失衡case预测/实测为`-18.40/-19.36%`，不支持新增generic swap correction；保留原v8 calibration。该轮为HEAD `935d643`加未提交Python/Lab改动的direct-sync provisional run，不是clean-commit paper artifact；生产quick、Plan V2、kernel、ABI与默认dispatch均不变。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_pairwise_ordering_mvp_20260902.md`。 |
| 2026-09-02 | v1.49 | hardware shortlist新增真实affected-lane与placed-event context contract：保存前后完整task/route序列、isolated lane load、head/tail暴露、phase/team dilation、cohort transition及critical lane/task，不保存raw event log。same-v8 high-skew复测的三个cross-lane swap把`62/68`-route的1T tail换到peer lane head，模型affected-tail增加`4.02--5.70 ms`且硬件退化`12.61--15.17%`；无新增tail的swap为`+0.40%`且区间跨零，四者均无placed critical-lane/expert switch。独立`68-route tail <-> 1-route head` holdout复现物理slowdown，但background/isolated模型residual仅`+1.95/-0.17`个百分点，五个tail/head case绝对residual均小于`1.96`点，未复现real-trace剩余误差。按predeclared gate不增加新物理参数、不改冻结v8 calibration，继续细化完整real-trace event context。 |
| 2026-09-03 | v1.50 | 独立分解固定1-route、1T expert的context bias；五个Plan V2执行完全相同tasks/routes/weights/total work，仅用首任务依赖将被禁用背景移到target lane结束之后。两次Arm 80C、31-round session中，full-cohort lane head相对isolated增加`0.823/0.881 ms`，其中16T background贡献`0.546/0.599 ms`、1T peers贡献`0.272/0.311 ms`，非加性交互仅`-0.011/-0.037 ms`。同一expert放在68-route前驱后减少`0.557/0.545 ms`，差异几乎全部来自W13；方向在paired P10/P90及独立session均稳定。冻结v8把1T-only预测为与isolated完全相同，并将full-head预测为`0.613 ms`而硬件为`1.475--1.482 ms`，独立复现了cold-W13 lane-head/background context缺口。但单个`M=1`点不足以识别新参数，因此不改公式或calibration；下一门槛是预声明`M={1,2,5,6,12}`route sweep及real-trace critical-switch holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_small_expert_context_20260903.md`。 |
| 2026-09-03 | v1.51 | small-expert probe升级为显式strict-DRAM协议，同时保留4-copy measured rotation：每个sample前关闭trace，在第五份地址不重叠的约660 MiB packed copy上同步执行完整`full_head` scrub，返回形成barrier后才运行measured mode；scrub不进入target span或trace。两次31-round scrub session保持原分解：full-head相对isolated `+0.811/+0.819 ms`，16T background `+0.558/+0.559 ms`，1T peers `+0.272/+0.270 ms`，interaction `-0.022/-0.022 ms`，after-68 `-0.523/-0.531 ms`。按target packed-weight虚拟地址过滤的Arm SPE高延迟样本中，W13 L3-hit incidence由pre-scrub约`6.9%`降到`1.7%`（356 miss/6 hit）；该比例受30-cycle采样门槛影响，不作为字节占比。结论升级为以DRAM cold-weight来源为默认，LLC residency视为benchmark干扰；模型与calibration仍未修改。 |
| 2026-09-03 | v1.52 | 完成预声明的scrubbed `M={1,2,5,6,12}` small-expert route sweep：full-head相对isolated增量为`0.811/0.836/0.621/0.655/0.283 ms`，1T-only增量为`0.272/0.302/0.113/0.128/0.022 ms`，证明victim敏感度随M增大快速衰减。trace同时证明target W13先与peer gather、后与peer W13重叠。解析模型增加默认关闭的显式gather-pressure calibration：`T_g=f_g+r_g ceil(M/t)`、`Q_g=alpha_g M H(b_in+b_pack)`，并把旧pre-W13 residual按stage时间比例移回compute phase；旧profile行为不变，schema/name升为9/v7。in-sample扫描`alpha_g≈3`使25点mean absolute relative error从21.7%降至12.6%，但M1的1T-only/after-68仍分别过估到`1.254/1.171 ms`（实测`0.907/0.921 ms`），因此不冻结calibration、不接入VND/LNS；下一步必须分离domain-local latency-bound gather与rank-wide streaming pressure。 |
| 2026-09-03 | v1.53 | 独立cross-LLC placement/phase probe选择每LLC-domain memory-injection作为首要缺失资源：固定1-route 1T target与15个68-route 1T aggressor，两次31-round session的local-minus-remote head差为`0.244/0.248 ms`，after-1纯W13窗口差为`0.225/0.233 ms`，四个paired P10均大于`0.21 ms`；remote相对isolated无稳定正惩罚。解析event模型增加默认关闭的domain cap `C_d=min(C_R(n_d), beta*C_R_sat/|D|)`，task DRAM dilation取rank与所触及domain最大值，explanation保存domain injection细节；旧profile保持rank-only，schema/name升为10/v8。`beta=0.76`把两项locality contrast mean absolute error由约`0.230 ms`降到`0.055 ms`，但叠加冻结v8 residual后旧25点route-context holdout MAPE由`21.7%`恶化到约`28.5%`，故只接受结构、不冻结参数、不接入VND/LNS；下一步联合替换已双计数的wide/narrow residual。 |
| 2026-09-03 | v1.54 | 完成严格fit/holdout隔离的phase re-accounting。新Arm fit corpus用不相交route `{3,4,7,8,10,16,24,48,68,600,1800}`覆盖全部`1/2/4/8/16/32/40/80T`，按stage envelope分别拟合`T_g=max(G_t,r_t ceil(M/t))`与`T_s=max(F_st,gamma_st T_physical)`；旧whole-total fixed/route、wide-team和narrow-team residual归零。独立phase repeat的gather/W13/W2/total MAPE为`16.13/3.49/4.70/4.38%`，旧route isolated holdout的W13/W2/total为`1.46/9.18/3.69%`。cross-LLC after/head差分拟合得到domain scale `0.787`和gather coupling `22.26`，但该差分无法识别absolute pressure：冻结后的25点并发holdout MAPE由v8 `21.72%`恶化到`273.97%`，48个可分辨pair错2个；real-trace high/median/uniformish Spearman为`0.345/0.075/0.368`，high top-8漏best，high/median分别有`10/6` false dominance。接受phase floor/scale结构，拒绝完整candidate及其domain/gather参数，不替换v8、不接VND/LNS；下一probe必须联合absolute slowdown、locality contrast和aggressor-count sweep。 |
| 2026-09-04 | v1.55 | 用锁定的aggressor-count probe做session-1联合absolute+contrast拟合，session-2只validation，不读holdout。重放已接受phase floor/scale，wide/narrow与whole-expert overhead保持identity/zero；loss为两类MAE之和，`alpha_g`上界8。最优点`beta=0.78`、`alpha_g=0.25`贴搜索下界，fit/validation联合MAE为`0.689/0.719 ms`，739个近优点、session-2对比残差系统为正，判定不可辨识。即使最小gather coupling，cross-LLC n15 head仍被预测为`+0.893 ms`（实测`+0.043 ms`）；关闭domain cap或把`alpha_g`从0.25提到1.00几乎不改该共模，说明主导项是rank `dram_bytes`对重叠68-route流的共享dilation，而不是9.44/9.45的两个标量。公式与production schema未改；拒绝冻结candidate，不替换v8、不接VND/LNS。下一缺口是DRAM contention作用域与n≈4饱和，而不是再估`(beta, alpha_g)`。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_absolute_pressure_joint_fit_20260904.md`。 |
| 2026-09-04 | v1.56 | 在锁定session-1 DAG上做rank DRAM vs domain-only离线消融，不改公式、不搜索参数、不读holdout。关掉rank DRAM dilation只影响placed allocator的第一次`dram_bytes`容量，domain cap仍用原来的`C_R(n_d)`。rank-only的cross n15 head为`+0.893 ms`，去掉rank DRAM后为`+0.454 ms`，只去掉约49%远程共模；剩余由`llc_bytes` dilation约`2.44`主导。domain-only不能把远程打到近零，也不能在n≈4饱和，`beta=0.20`仍把same n15推到`+5.71 ms`。不新增default-off结构。下一缺口是同一DAG上的LLC作用域与victim是否不该继承68-route peer的GEMM dilation。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_rank_dram_domain_scope_20260904.md`。 |
| 2026-09-04 | v1.57 | 在同一锁定DAG上做rank LLC vs domain LLC离线消融，不改公式。关掉rank LLC只对传入全部domain id的`llc_bytes`调用返回无穷，保留单key domain LLC；全部臂保持rank DRAM关闭。rank LLC解释剩余远程共模的100%（cross n15 `+0.454`→`0 ms`），same-LLC曲线不变。domain LLC仍随count增长（n2/4/8/15为`0.168/0.397/0.447/0.590 ms`），达不到n≈4饱和。去掉全部LLC后本地n15仅`+0.001 ms`，再关L2不变；模型victim已近似isolated，硬件same-LLC n15仍为`+0.312 ms`。不新增结构。下一缺口是victim-asymmetric saturating same-LLC tax。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_rank_llc_domain_scope_20260904.md`。 |
| 2026-09-04 | v1.58 | 在同一锁定DAG上做victim-asymmetric dilation消融，不改公式、不搜索参数。对称臂仍把cohort DRAM/L2/LLC dilation套到每个task。`compute_bound_skip`与对称臂相同，因为1-route 1T W13是transfer-bound。`same_llc_peers`把远程打到0、本地曲线不变。`own_demand`把n15打到`+0.001/0 ms`。没有任何臂同时满足远程近零、n4 contrast与n≈4平台。不新增结构。过预测来自短transfer-bound victim继承68-route字节dilation；硬件留下saturating same-LLC occupancy税，不能在本count sweep上拟合。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_victim_asymmetric_dilation_20260904.md`。 |
| 2026-09-04 | v1.59 | 增加独立于count sweep的aggressor-$M$ occupancy probe，不改公式。victim仍是1-route 1T；count固定0/1/4；aggressor $M\in\{1,4,16,68\}$。Arm session-1 overlap有效、远程近零；same-LLC n4 head为$M=1/4/16/68$的$+0.660/+0.639/+0.402/+0.290 ms$，随$M$下降。预声明occupancy/duration/utilization三态都不成立。不新增结构。leftover是同LLC上并发transfer-bound流的争用，不是68-route字节利用率。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_aggressor_m_occupancy_20260904.md`。 |
| 2026-09-04 | v1.60 | 增加fill-port vs stream probe，不改公式。同一16个LLC7核（logical `48-63`）上对照1×16T $M=1$与16×1T $M=1$。Arm session-1 overlap有效、远程近零；same-LLC head为1T/1×16T/16×1T的$+0.018/+0.014/+0.160 ms$。`fill_ports`不成立；`one_stream`方向成立但many$-$wide=$0.146<0.20$，预声明为`inconclusive`。1×16T $M=1$是一条packed-B，W13 overlap core-ms为$1.08$对many的$10.58$。该切片上16×1T远小于occupancy在victim邻域`65-79`的n4 $M=1$ `$+0.660 ms$。不新增结构、不读holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_fill_port_vs_stream_20260904.md`。 |
| 2026-09-04 | v1.61 | 增加stream-count composition probe，不改公式。等线程阶梯在`48-63`上对照1×16T/2×8T/4×4T/16×1T与4×1T；`8+8+4+1`放在`43-63`对照同起点4×1T与21×1T。Arm session-1 overlap有效；阶梯same-LLC head为$+0.003/+0.035/+0.075/+0.070/+0.155 ms$；mix/四起点1T/21×1T为$+0.110/+0.103/+0.301 ms$。自动化签名`stream_count`，`thread_count`不成立。21×1T远程$+0.211 ms$，四流mix远程$+0.042$。不新增结构、不读holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_stream_count_composition_20260904.md`。 |
| 2026-09-04 | v1.62 | 增加unified weight-block probe，不改公式。把一层W13+W2拷进同一匿名allocation的两个contiguous view，对照现有两块packed tensor。问16×1T leftover是否因此下降；这不能把16条并发专家流变成1条流。本机未预留HugeTLB。Arm session-1 overlap有效；16×1T leftover split/unified为$+0.153/+0.157 ms$，差$0.004$，签名`layout_neutral`。1×16T为$+0.041/+0.042$。远程16×1T为$+0.088/+0.098$。本probe不加结构、不读holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_unified_weight_block_20260904.md`。 |
| 2026-09-04 | v1.63 | 增加weight THP probe，不改公式。同一W13+W2块上对照mmap+MADV_NOHUGEPAGE与mmap+MADV_HUGEPAGE，用smaps AnonHugePages验证。问2 MiB页是否降低16×1T leftover。不改FUSED_CPP_PAGES，不预留HugeTLB。Arm session-1 overlap有效且页验证通过：4 KiB AnonHugePages为0，THP覆盖100%。16×1T leftover small/THP为$+0.167/+0.162 ms$，差$0.005$，签名`page_neutral`。1×16T为$+0.006/+0.008$。isolated span为$0.577/0.571 ms$。远程16×1T为$+0.145/+0.143$。本probe不加结构、不读holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_weight_thp_20260904.md`。 |
| 2026-09-04 | v1.64 | 增加stream-pressure PMU分账，不改公式。PMU-only DAG只保留victim与实际并发peer，排除旧等工作量协议中victim后的delayed background；初始化、warmup与18-expert不相交packed-copy scrub均在counter disabled时完成。两个嵌套perf FIFO在同一31-run cell同步采CPU304 core、LLC7十个L3C slice、NUMA3八个DDRC与native trace。session-3在isolated/1x16T/4x4T/4x1T/16x1T上测得victim span $0.477/0.517/0.627/0.644/0.704 ms$，DDRC read-command latency $32.6/43.9/55.8/49.2/57.0$ cycles，DRAM read traffic $10.0/21.5/37.0/37.5/108.3 MiB/call$。queue latency、victim LL-read miss与backend-stall在约四份互异B后趋于平台，而aggregate bytes/bandwidth继续增长；排除bytes线性模型，识别为victim-visible LLC miss加DDRC排队。4x1T跨session仍漂移$0.048 ms$，故五点不足以识别公式，不新增结构、不读holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_pmu_20260904.md`。 |
| 2026-09-04 | v1.65 | 完成固定16 peer线程的$0/1/2/4/8/16$互异packed-B PMU sweep及独立4x1T repeat，不改公式。全部cell以嵌套perf FIFO同步CPU304 core、LLC7 L3C和NUMA3 DDRC，event running 100%、overlap精确。过isolated原点的queue/LLC单变量LOCO MAE为$0.0453/0.0364 ms$、RMSE为$0.0464/0.0459 ms$、最大误差为$0.0574/0.0767 ms$。LLC平均误差虽低，但count-1 pressure为负；queue保持非负并在独立4x1T repeat仅误差$-0.0010 ms$，但低估count 8/16。允许intercept的敏感性不改变结论。两者均不冻结；queue仅作为同一长生命周期process内paired counter-reset实验的下一候选，LLC只作guardrail。不读旧holdout、不替换v8。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_count_loco_20260904.md`。 |
| 2026-09-04 | v1.66 | 完成同进程、同packed allocations、随机交错isolated/candidate及逐cell直接`perf_event_open` reset/read的paired PMU。主session一次打开60个CPU304 core、LLC7 L3C、NUMA3 DDRC event，31轮全部running ratio 1、overlap精确。count $0/1/2/4/8/16$ paired slowdown为$0/+0.0082/+0.0295/+0.0455/+0.2040/+0.2042 ms$，queue pressure为$0/+11.1/+19.0/+21.0/+41.8/+36.0$ cycles；queue在全部正count的P10为正，LLC miss pressure在1/2/4为负。paired queue/LLC anchored LOCO MAE为$0.0515/0.0778 ms$，affine MAE为$0.0355/0.0432 ms$。独立4x1T repeat slowdown P10/median/P90为$+0.0329/+0.0878/+0.1374 ms$，queue模型仍低估$0.0283 ms$。接受测量协议，拒绝LLC standalone与queue线性模型；不得从本session加knee/threshold/residual。不读旧holdout、不替换v8。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_paired_pmu_20260904.md`。 |
| 2026-09-04 | v1.67 | 固定packed-B count与expert起点，完成两次同进程paired request-shape分解，不改公式。count 4对照start-aligned 4x4T/4x2T/4x1T：变窄稳定降低queue，但全部victim span pair至少一轮区间跨零，未识别width latency效应。count 8的8x1T-8x2T在两轮victim span为$-0.1133[-0.1676,-0.0815]/-0.0775[-0.1411,-0.0247] ms$，queue为$-16.90[-30.68,-7.65]/-14.06[-20.61,-7.19]$ cycles，接受`narrower_faster_with_lower_queue`。拒绝narrow-team asymmetric harm。压力至少依赖互异B count与team-width/issuer shape的交互；只有两个count，禁止拟合knee、width multiplier或residual。不读旧holdout、不替换v8。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_request_shape_20260904.md`。 |
| 2026-09-04 | v1.68 | 完成预声明的count $\{4,6,8\}\times$ width $\{1,2\}$结构proxy网格，count 6全锁holdout，不改公式。两次同进程31-round session中count-6 1T/2T slowdown为$0.0685/0.1166$与$0.0650/0.1103 ms$。以count 4/8拟合的distinct-B、requester threads、product、measured-W13-overlap oracle全部失败；distinct-B不能区分width，其余虽方向正确但proxy-to-queue参数漂移$25.8--32.3\%$并至少失败一个queue/slowdown gate，oracle仅首轮通过。queue-to-slowdown slope为$0.00440/0.00476 ms/cycle$，漂移仅$7.5\%$，故缺口在plan geometry到controller queue state而非victim系数。最终`accepted=[]`、`stop_absolute_model_expansion=true`；禁止再加knee/width/queue/residual，保持frozen v8，转向partial order、top-K recall与false pruning。不读旧holdout。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_proxy_grid_20260904.md`。 |
| 2026-09-04 | v1.69 | 将anchor-relative partial order接入离线shortlist与可复用best-improvement闭环，不改event mean、production planner、Plan V2或runtime。只有完整区间低于anchor的candidate可做dominance pruning；incomparable保留，因budget未入选单列为`budget_deferred`；只有完整区间越过2% actionable margin才更新incumbent。neighborhood audit仅接受带zero-false-pruning、对应top-K measured-best recall及匹配calibration/extension identity的report，默认top-16。same-v8跨session 41-pair replay仍为41/41 incomparable、0 false pruning且top-8/16/32均保留三条trace measured best；混合历史129-pair反例replay为0 false pruning，high-skew `cp_sat_06`从point-model错误方向保留为incomparable，但top-8漏3个measured best、top-16全部保留。因此完成保守接线和gate，尚未证明有效decision coverage或multi-start VND收益。 |
| 2026-09-04 | v1.70 | 完成partial-order驱动的四起点model-only VND基线。runner从full、one-step、greedy与最佳homogeneous fixed-width strict plan独立出发，每轮重新计算critical/random neighborhood、canonical去重、三向关系、top-16 frontier、event/context calls、wall time与best-so-far；只有lower gain bound严格超过2%才更新可执行incumbent。若report没有任何placed-context key，则样本不足operator直接回退family，不运行必然miss的candidate event explanation。Arm NUMA3三trace共12个命名start、7326个unique candidate、7350次event call和1067.58 s；关系为0 better、1002 worse、6324 incomparable，全部iteration-0停止且0 accepted。high-skew greedy最接近门槛的point/lower为3.448/1.679%，仍不可接受。历史`cp_sat_06`至少与full相差53个expert-width assignment，且所有high-skew start没有第一步，故单调VND不能恢复该机会。保留partial order做top-16与one-sided pruning，停止当前单调VND；下一搜索实验转向template-level LNS/non-monotone global proposals。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_partial_order_vnd_model_replay_20260904.md`。 |
| 2026-09-04 | v1.71 | 将high-skew四起点top-16 incomparable frontier冻结为自包含canonical state+PlanV2 bridge，并为每起点加入upper-bound最近与最远的两个`candidate_worse` sentinel；builder逐项复算canonical hash并跨起点去重，得到4 anchor、64 incomparable、8 sentinel，共76个plan。两次独立Arm NUMA3 session各用单进程、同批4-copy weights、5 warmup、31 randomized paired rounds；session内stable >2%分别6/5个，跨session同一pair稳定者2个。fixed-width lane-head相邻交换（1341-route与261-route expert）model仅预测+0.014%，硬件median为+4.690/+3.988%，P10为+3.029/+2.022%；greedy domain-local `16T->8T+8T` split预测+2.319%、lower +1.181%，硬件median为+3.819/+3.376%，P10为+2.503/+1.965%。强制保留的model-best-lower greedy候选仅+1.461/+0.081%且P10为负。8个worse sentinel无跨session false pruning。按预声明决策，局部邻域有真实价值而comparator positive coverage不足；下一步采用partial-order pruning+top-K双session硬件rerank的beam search，暂不直接进入template LNS，不从本frontier重拟合radius。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_partial_order_hardware_frontier_20260904.md`。 |
| 2026-09-04 | v1.72 | 完成修正后的hardware-assisted beam depth-2/3。首次depth-2只扩展相对各自较差anchor稳定的两个winner，后续发现两次absolute best均为full分支的`4115ab99...`，故该轮排除；analyzer新增跨session normalized absolute elite，beam改为同时扩展global measured incumbent。修正depth-2从3 parent生成2064 unique candidate（0 better/238 worse/1826 incomparable），61-plan双session找到`514ced0d...`两次absolute fastest `31.808/31.721 ms`；其相对immediate parent仅+0.745/+1.055%且P10为负，但相对original full为+2.146/+3.228%、P10 +0.146/+1.532%，证明per-edge 2% acceptance与global elite retention必须分离。depth-3从共识absolute top-2生成1461 candidate（0/199/1262）并测45 plans；两次均0 stable >2%，absolute fastest不同，共识`0ffab6b9...`相对parent仅+1.411/+0.921%且P10为负。一个worse sentinel偶为session-2 absolute best，但paired为-0.774/+0.039%，不构成false pruning并暴露sub-percent rank噪声。按预设max depth 3停止local beam，保留full、`4115...`、`514ced...`，下一步转template-level LNS并在等event/hardware预算下比较；不降低margin、不重拟合radius。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_hardware_assisted_beam_20260904.md`。 |
| 2026-09-04 | v1.73 | 增加lane-atomic template-level LNS，不改变production planner、Plan V2、kernel、schema、ABI或冻结v8。对critical/random expert和target destroy `4/8/16`选择最小连续lane closure，分别生成domain-local与cross-domain block；cross repair只允许domain-contained新lane。每块从已校准宽度的合法partition中保留4个near/far template，以isolated-load placement beam `16/32/64`联合修复width、assignment和四种temporal order。high-skew从full、`4115...`、`514ced...`各做两个restart，在3,512次event call和106-plan硬件预算下与已关闭local beam的3,535次/106-plan对齐；LNS模型用时1,193.47 s，对照beam 507.06 s。两次独立31-round NUMA3 session得到42个stable comparison、40个unique winner、0/11 false-pruning；所有winner均为cross-domain。共识`189d70b0...`绝对median `31.249/31.297 ms`，相对`4115...` paired median `+3.880/+3.271%`、P10 `+1.974/+1.506%`，相对保留的`514ced...`绝对median比值收益`+3.961/+3.528%`。接受high-skew template-LNS结构并继续硬件rerank；median/uniformish仍是未跑holdout，禁止三trace或production结论。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_template_lns_20260904.md`。 |
| 2026-09-04 | v1.74 | 完成template-LNS high-skew第二层与median/uniformish冻结mixture复验，并将LNS partial order降为diagnostic-only。high-skew depth-2用2,351 event call/72 plan，只剩1个strict stable candidate，停止第三层。median用4,659 call/145 plan，winner `2ab435...`为`31.381/31.393 ms`，相对strongest full anchor `+2.671/+2.863%`；uniformish用2,345 call/74 plan，winner `50a7d4...`为`31.517/31.698 ms`，相对strongest control `+2.057/+3.250%`。三trace固定LNS neighborhood均通过两场strongest-anchor 2% gate。但median有2个cross-session false-pruning sentinel，absolute best被模型以`-3.770%`、upper `-2.001%`错误判为worse；uniformish有1/26 model-better未通过strict P10复验。因此LNS runner设`Prune=Accept=empty`：relation仅诊断，未测项为budget-deferred，硬件frontier区分`model_better_frontier/model_worse_spectrum`并返回跨session共识elite。local VND comparator、frozen v8 mean、production planner、Plan V2、kernel、ABI和runtime不变。suite总计12,867 event call、3,488.51 s model wall、397 frontier plan与8个hardware session。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_template_lns_suite_20260904.md`。 |
| 2026-09-04 | v1.75 | 为template-LNS增加relation-agnostic categorical farthest-first shortlist，不改production planner、Plan V2、kernel、ABI、冻结v8或local VND comparator。候选特征只用operator、target destroy、actual closure、width histogram、LLC-domain assignment和同一proposal pool内的score quantile；relation、残差半径和硬件时间不进入选择。每start top-16是top-32 audit前缀，落选一律`budget_deferred`。冻结measured-suite设计回放在K=16上四条case均保住绝对实测最好与共识winner，selected-best regret为0，三次打乱输入顺序hash不变；K=8/12在high-skew L2与uniformish上仍有regret。政策`relation_agnostic_categorical_farthest_first_v1` rule SHA256为`6c07e7b9...`。独立median holdout未跑，不得adopt进production。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_lns_diverse_shortlist_replay_20260904.md`。 |
| 2026-09-05 | v1.76 | 在Arm-codex NUMA3上冻结selector v1后打开独立median frontier：route layer 4、proposal seed `20261010`、每unique parent一次restart、shortlist 16/audit 32。4个reconstructed control的canonical hash与旧median VND一致。模型2,330次event call、797.01 s，冻结132-plan frontier；两场31-round session各约168 s，bit-exact。Selector nested-recall全部通过，K=8到K=32的selected-best regret均为0。共识`7cac2afd...`为`31.720 ms`量级的cross-domain d8、48-expert closure，模型预测`-0.087%`且incomparable。相对strongest full `98a32da5...`的绝对median收益为`+2.579/+1.916%`，第二场未过2%，故neighborhood/proposal失败与selector召回成功分开记录。采用v1作离线LNS shortlist，不改K、不改selector、不进production。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_lns_diverse_independent_median_20260905.md`。 |
| 2026-09-05 | v1.77 | 为template-LNS补齐search-cost split，并对确认的enumeration热点`_beam_assign_tasks`做一轮等价加速：预计算`(expert,width)` retarget与增量signature，不改last-write-wins、placement_priority或四种顺序策略。隔离profiler上full/one-step的beam为7.045→2.245 s与67.573→17.221 s，计数不变。冻结median seed `20261010`生产路径墙钟797.01→687.55 s；exact 341.43 s、diagnostic shortlist 278.77 s、sample 56.31 s（beam 40.56 s）。候选hash、抽样、模型分数、quantile与有序top-16/top-32与冻结artifact一致，`equal=true`。event simulator与跨lane增量回放未改。峰值RSS 309,092 KB，无797 s基线RSS。下一步才是等预算multi-restart。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_lns_search_breakdown_beam_equiv_20260905.md`。 |
| 2026-09-05 | v1.78 | 冻结selector v1、K=16（按unique parent池化）和现有operator mixture，在exact cap 2,400与硬件85-plan（4 reconstructed + elite `2ab43572...` + 64 top-16 + 16层外抽查）下比较1 restart（N=50）与2 restart（N=25）。两边都选出`0418b884...`，两场相对full均超过2%。2-restart相对elite两场为正，1-restart第二场为负；K=16集合只重叠15/64；层外16个样本未打过selected。搜索墙钟558.71/683.67 s，实际event call 2384/2330（八个start多付parent exact）。离线median采用2 restart+N=25，不改K、不改selector、不进production。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_lns_restart_budget_20260905.md`。 |
| 2026-09-05 | v1.79 | 冻结selector v1、K=16、2 restart、N=25和exact cap 2,400，将proposal seed从`20261010`换成`20261011`。搜索2,378 event / 2,362 unique / 550.90 s，未生成`0418b884...`；K=16与上一seed只重叠3/64。两场selected-best为`2fd99a11...` / `8deba718...`，相对full `−0.018% / +0.371%`，未过2%，也未快于elite或注入的上一seed winner。`0418b884...`作为`previous_seed_selected`对照在session 1仍是实测最快（`+3.302%` vs full）。这是proposal-seed失败，不改分配、不改selector、不进production。完整记录见`optimizations/fused_moe_sve/results/arm_codex_80c_lns_second_proposal_seed_20260905.md`。 |
| 2026-09-06 | v1.80 | 在冻结2-restart N=25路径上审计五个实测快plan的丢失阶段，不改sampler、selector、K或production。full parent `98a32da5...` 的 `lns_00_r00`/`lns_00_r01` 枚举表明五者都可达；`0418b884...`/`7cac2afd...`/`1faf090a...` 的seed间丢失是operator内shuffle-truncate，不是预采样缺失；`2ab43572...` 进入scoring后排在parent top32之外。Lab候选 `structural_coverage_then_random_v1`（`275dd663...`）与baseline同shuffle，design replay不恢复sampling-lost hash，不接入生产抽样，不开Task 5硬件。完整记录见`optimizations/fused_moe_sve/results/template_lns_cursor_todo.md`。 |
| 2026-09-06 | v1.81 | Lab候选 `structural_coverage_closure_width_v1`（`e5a35b06...`）在同一shuffle上按`(closure_bin, width_histogram)`做coverage。冻结模型 `9b334b78...`/`cc53d43d...` 的full-parent回放相对N=25 shuffle-truncate：seed `20261010` 丢掉已抽中的 `0418b884...`，五个tracked hash中没有任何sampling-lost成员被恢复；`(bin,hist)` key覆盖在24/24 operator cell上升。判定reject，不改 `sample_template_lns_neighborhood`、selector、K，不开Task 5。完整记录见`optimizations/fused_moe_sve/results/template_lns_cursor_todo.md`。 |
| 2026-09-06 | v1.82 | 冻结full-parent critical cross-domain d4 参考池：shipped枚举、全局canonical去重后再滤该operator。四个保存的critical输入hash集合相同，unique 452，digest `a10eeba6...`。`0418b884...`与`2ab43572...`均在池内；加四个reconstructed control后去重为456。历史session只测到11/452。诊断预算约541 s/场、两场约1083 s，超出既有80 LNS-candidate槽，不改N=25/K=16/64+16。Task 6B未授权，Task 5关闭。完整记录见`optimizations/fused_moe_sve/results/template_lns_cursor_todo.md`。 |
| 2026-09-06 | v1.83 | 在已拒绝的 count-6 proxy 网格上做事后 planner-visible geometry：\(n_B\sqrt{t}\)、冻结 v8 isolated peer W13/operator core-ms，以及 \(a n_B+b n_{\text{threads}}\)，不用实测 overlap。原门槛下全部失败，斜率漂移 \(26.9\%\)--\(62.4\%\)；同一 8x2T plan 的 queue 为 \(48.17/29.14\) cycle。不改公式、不读旧 holdout、不替换 v8。完整记录见 `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_plan_geometry_proxies_20260905.md`。 |
| 2026-09-06 | v1.84 | 按原 proxy-grid 协议在 Arm NUMA3 加测三场（seed `20260926/27/28`）。8x2T 配对 queue 中位为 \(12.58/32.19/27.89\) cycle，与历史 \(48.17/29.14\) 一起跨度 \(12.58\)--\(48.17\)。三场均非独占（`bench_meformer_`、`tokio-rt-worker`）。不改公式、不冻 queue、不替换 v8。完整记录见 `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_queue_repeat_20260906.md`。 |
| 2026-09-06 | v1.85 | production quick 齐次宽度比较改用已校准 wide-team occupancy 代理：LPT packing 仍用孤立 $T_{iso}$，makespan 乘 $B_t+(S_t-B_t)q$，与 placed event 的 peer fraction 一致。冻结 v8 表已含 $B_t/S_t$，此前只在 heavy event 生效。不改 Plan V2、kernel、ABI、$T_{iso}$ 公式或 LNS。无表时系数为 1。这只针对过宽齐次队，不恢复局部邻居序。 |
| 2026-09-06 | v1.86 | Arm NUMA3 5 warmup/31 pair/4 copy 复测 quick 对 fixed 8T。active-set-8/16 不再选 40T，配对 `-0.03%/+0.30%`，关闭原 40T 回退；active-set-64/128 与 tiered 过窄到 20x4T，稳定亏损 `11.90/10.92/14.25%`；bimodal 仍为 5x16T 且 `+30.26%`。quick 仍不支配 fixed 8T。完整记录见 `optimizations/fused_moe_sve/results/arm_codex_80c_quick_vs_fixed8_wide_team_20260906.md`。 |
| 2026-09-07 | v1.87 | Lab workspace 阶段候选清零 operator residual，独立拟合 gather/W13/W2；第一场拟合、第二场重复，M1 不拟合。拟合点阶段误差下降但 M1/1T 外推失败，拒绝采用，2/4T 不外推。156 个完整 placed 消融定位 wide-team pressure 为大 M/16T 过度放大的主要模型来源，同时保留 narrow/DRAM 的小 M/1T 反例。v8、production、搜索域和剪枝不变。报告：`optimizations/fused_moe_sve/results/workspace_phase_reaccount_ablation_20260907.md`。 |
| 2026-09-07 | v1.88 | 完成36-cell 小 M/2T/4T isolated 网格与6场3126-call校验；修复中间 task 移到 head 后旧后继未接回前驱的 Lab 依赖错误，失败数据不参与分析。M7 不拟合，M1 保护保留。Affine GEMM 通过已测 M1 guard 与 M7 GEMM 20%检查，硬 floor 仍高估 M1/1T；但 gather 留出失败且部分 intercept 跨场不稳定，整套候选不合格，不替换 wide/narrow，不改 production/schema/剪枝。报告：`optimizations/fused_moe_sve/results/workspace_floor_identification_20260907.md`。 |
| 2026-09-07 | v1.89 | 完成冻结 v8 A/B/C 账单与28次 workspace placed 回放。A 已按 owner 计费，M1341/16T steady B 下层 refill 已为0；M1 A payload 与 cache-line footprint 不同，L2-hit delivery 与 LLC refill 尚需独立核对。Lab 容量安全 lifetime 分配虽将 compute-end MAPE20.24%降至13.01%，但 M1/1T 阶段 MAPE38.79%恶化到66.72%，且 median elite 顺序仍错，拒绝替换。保留有界诊断参考，不改 production、schema、校准或剪枝。报告：`optimizations/fused_moe_sve/results/ab_supply_ledger_overlap_20260907.md`。 |
| 2026-09-07 | v1.90 | Lab区分A payload/line coverage/refill估计与未识别实际A/B refill，核对B-only endpoint探针；独立私有L2供给下界与LLC/DRAM refill，保留原burst服务，不用phase平均。35形状与42次workspace回放中私有下界仍被compute隐藏；整体MAPE20.24%→20.39%、M1阶段38.79%→38.71%，排序未改善，不采用。旧probe的共享B与无逐轮scrub条件限制仍显式保留，不改v8、production/native/schema/剪枝。报告：`optimizations/fused_moe_sve/results/burst_private_endpoint_20260907.md`。 |
| 2026-09-08 | v1.91 | 将已有isolated证据限定到1/2/4T的29形状、87阶段点，复现无operator residual的独立affine候选，启动/compute/暴露供给单列，不引入并发。M1 W13/W2平均误差降至3.28%/3.72%，但M7 W13恶化至10.13%；gather历史验证仍失败。M1/2T两场gather中位13/29us对任意固定预测的观测误差下界38.52%，相同ceil特征亦有冲突。只保留stage-only估计器，拒绝T_iso/planner导出；不换校准、不部署。报告：`optimizations/fused_moe_sve/results/small_t_isolated_stages_20260908.md`。 |
| 2026-09-08 | v1.92 | Lab exact-M W13×1/2/4T两场99-cell网格，第一场peers0拟合条件B-only响应，两场peers4/8验证MAPE2.008%，同缓存lookup2.252%，仅3/12门槛通过。冷isolated同shape跨场lookup误差0.437%对v8 7.499%；固定计算/访存占比未被识别，保留未见M基准验证方向，不扩W2、不部署、不改剪枝。报告：`optimizations/fused_moe_sve/results/kernel_joint_response_20260908.md`。 |
| 2026-09-08 | v1.93 | Lab M1/M12访存压力曲线：0–16 reader、1/2/4T、两场84-cell×31轮。最高档DDRC读流量72–76GB/s、1T B-only变慢25.6%；M1约变慢33/13/10%，M12仅约1–1.7%，未覆盖两场均>2%的转折。区分实测小幅损失、操作门槛与未知物理瓶颈；不拟合拐点、不改模型或剪枝。报告：`optimizations/fused_moe_sve/results/memory_pressure_curve_20260908.md`。 |
| 2026-09-08 | v1.94 | Lab条件供给响应：第一场拟合与六档LOPO选型、冻结双anchor预测第二场，六组均选择linear。M1第二场MAPE0.896%/max1.871%，M12为0.140%/0.690%，优于isolated常数。阈值收益不足或不稳定；明确历史回放、实测供给输入和轻微外推限制，不部署、不改剪枝。报告：`optimizations/fused_moe_sve/results/pressure_response_fit_20260908.md`。 |
| 2026-09-08 | v1.95 | 冻结响应采集新3/6/10/14 reader两场，M1主验证MAPE0.749%/max2.347%，M12为0.194%/0.445%；11/12 shape/session门槛通过，首场M12/1T略劣于isolated基线。0/16控制不混入主评分，不重设双anchor、不拟合新数据、不改v8或剪枝。报告：`optimizations/fused_moe_sve/results/pressure_response_prospective_20260908.md`。 |
| 2026-09-08 | v1.96 | 冻结响应换真实W13后台，两场60-cell×31轮，reader/无后台对照全过，真实背景仅5/12门槛通过。M1平均误差2.027%/max7.998%，M12平均误差0.664%劣于isolated0.247%；近等B-only的小/大后台造成M1/1T约8%配对差异。保留原模型及上下文反例，注明1T超拟合范围，不增加参数、不部署。报告：`optimizations/fused_moe_sve/results/real_kernel_background_20260908.md`。 |
| 2026-09-08 | v1.97 | Lab补M1/1T A-only与AB加载骨架，B-only机器码精确匹配生产。两场18-cell×31轮真实大/小后台差34.77/29.29us复现，但A-only基本不变、AB反向快15us；full-no-store仍慢38.90/33.57us。否定直接加正A供给项的本轮解释，保留计算/加载交互方向，不拟合或部署。报告：`optimizations/fused_moe_sve/results/ab_supply_contrast_20260908.md`。 |
| 2026-09-08 | v1.98 | Lab计算数量/依赖扫与无加载确认共4场。相同矩阵指令改用预置寄存器仍有后台差；去掉A/B加载差近0，换整数/NOP保持负差。额外backend等待与LLC路径变化相伴，局部定位为矩阵执行与访存共存损失，不声称唯一硬件端口/队列来源；模型冻结，无生产改动。报告：`optimizations/fused_moe_sve/results/m1_compute_issue_20260908.md`。 |
| 2026-09-08 | v1.99 | Lab L1热A4KiB/B8KiB环形探针两场39-cell×31轮全部通过命中事件比门槛；流式matrix4大/小后台差33.82/35.31us，热版本仅+0.62/-0.53us且区间跨零。固定指令组合不足以解释原后台差；热matrix2仍不完全重叠，matrix8快于所选pure控制，禁止据此拟合单一发射端口系数。保留缓存供给/执行交互方向，冻结模型、公式与剪枝。报告：`optimizations/fused_moe_sve/results/l1_matrix_mix_20260908.md`。 |
| 2026-09-08 | v1.100 | Lab实测分层供给候选仅用双LLC第一场same/other8/24/39拟合；训练count-LOCO选择queue+local+interaction。冻结预测第二场，留出16/32 MAPE/max1.54/4.18%，未训练balanced16/24/32为2.92/4.48%；高压力48/64/78外推11.70/15.57%失败。总体3.22%平均误差不豁免失败；输入为执行期间PMU，非plan-visible、非盲测或唯一物理分解。不重拟合holdout、不改生产v8/profile/剪枝。报告：`optimizations/fused_moe_sve/results/layered_measured_supply_20260908.md`。 |
| 2026-09-19 | v1.101 | Lab：Arm-codex n_tile16 满载窗口表候选 `ARM_CODEX_NUMA3_80C_TP4_F512_N16_V1`（v1.102 起为 V2）（含每宽度实测时间比 $r(t,M)$，未注册为默认）；`IntervalPlanner` 新增显式 opt-in `stage_window_policy`，有 `time_scales` 时 task cost 为 $T_{iso}r$，默认、Amazon 表、schema 与 ABI 不变。六个未见层验证 G1/G1w/G2 仅 3/2/1 of 6 通过且一层慢 4.6--6.8%，不改 planner 默认。 |
| 2026-09-19 | v1.102 | Lab：W13=1 tile 下的 W2 窗口满载扫描；W2 4--8 tile 只在 M 48--192（2T）/48--96（4T、8T）按冻结规则采用，M≥288 一律 W2 整条 stripe；候选表更名 V2，仍为 opt-in，默认不变。 |
| 2026-09-19 | v1.103 | Lab：窗口机制 PMU 验证。整条 stripe 在并发 stripe 超过 LLC 时约每个 12 行 M panel 从 DRAM 重读一次 B；W13 1-tile 窗口使 DRAM 读减少 27--75%，L2 refill 仅降 0--9%，收益在 LLC/DRAM 层；大 M 代价是每窗口重扫 A13。W2 窗口大 M 变慢与 DRAM 写增加相伴，而非 A2 重扫。无表或 planner 改动。 |
| 2026-09-19 | v1.104 | Lab：满载 DRAM 写归因。glibc 默认 + THP always 下，Plan V2 每次调用新分配的 FP32 `route_out` 每次缺页清零（M=720 写 22--29 GiB），常驻分配器配置无缺页且快 17--35%；jemalloc 默认会在调用间归还该内存，永不归还时等同常驻。此前满载实测与机器响应标定均包含此开销；模型公式与默认未改。 |
| 2026-09-19 | v1.105 | Lab：jemalloc never-purge 下重跑四个窗口实验（`tmp/jemalloc_rerun_20260919`）。候选表以 V3 替换 V2（`ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3`，同一构建流程）：2T/4T 在 M 24--720 均用 W13 1 tile，8T 扩到 M 24--480，16T 仅 M 24/48；glibc 期的 W2 大 M 惩罚与 M=24 双峰为缺页伪影。runtime_windows 结论保持（4T 窗口 vs 8T 全条带 −8.6%）；未见层验证 G1/G1w/G2 为 0/3/1 of 6，仍不改 planner 默认。V3 仍为 opt-in 未注册；公式与 schema 不变。 |
| 2026-09-19 | v1.106 | Calibration：jemalloc never-purge 下重测 v8 中受 `route_out` 缺页污染的三层（wide-team $B_t/S_t$、1T/2T `by_width`、窄 team 修正），得到 v9（`arm_codex_numa3_80c_jemalloc`）；$S_{8/16/40}$ 下降 9--16%，闸门层留出实测/预测中位 0.898→0.984。服务速率、公式、schema 与默认消费方不变。 |
| 2026-09-19 | v1.107 | Lab：探针校准的融合 event 模型 v10 候选。删除 v9 的 spill=1.0 DRAM 需求、DRAM 饱和稀释、$B_t/S_t$ 与窄 team 修正；孤立时间为 $(1+\varepsilon)\sum\tau_p+O(t)$（$\varepsilon=0.065$，$O=10+2.5t$ µs）；争用由四条探针实测曲线 $D_{LL},D_{LS},D_{SL},D_{SS}(t,n)$ 给出，phase 分装载/稳态，组合规则测量前冻结、无自由参数，整计划只作验证。实测争用集中在权重装载段之间（$D_{LL}$ 至 1.6--2.0，$D_{SS}\le1.03$）；单一硬件曲线的机理检验未通过，按宽度分档服务表报告。window 选择取 V3 表，时间尺度随模型超额在孤立收益与表值间线性过渡；$t_{over}=0.78$ ms 直接测得。开发整计划 regret@1 0--2.7%，wall/预测 1.03--1.10。production、Plan V2、默认消费方不变。M4 验收：W1 过、W2 差 0.003 个百分点未过（regret@1 最大 5.003%，@2 最大 3.4%）、S2 过、S1 未过（小 M 目标低估）；采用收窄声明，不开第二轮。 |
| 2026-09-20 | v1.108 | opt-in 接入：`ProbeEventModel`（v10 作为 planner cost model，与 Lab 数值一致）、标定资产 `probe_event_v10_20260919.json`、`IntervalPlanner` 的模型 quick 比例钩子、模型目标 LNS `model_lns.py`（relocate/swap/split/merge/复合 repack/reorder，lane 限于单 LLC 域）。默认消费方、native planner、Plan V2 不变。效果实验 E1 设计冻结于 `tmp/v10_integration_20260920/decision.md`。 |
| 2026-09-20 | v1.109 | Lab/opt-in：v11 标定（孤立校正、按 M 索引的稳态曲线、中 M 背景装载等价 $w(M)$，均由 P6 探针测得，组合规则冻结，无整计划拟合），`ProbeEventModel` 支持小数核计数与上述可选字段（v10 资产数值不变）；模型目标 LNS 增加 rebalance/recreate/阈值接受与多起点下降（P2 小实例全部达到精确最优）。E1 结果：v10 驱动搜索利用 2T 与中小 M 误差；E2 在新层上验证 v11。 |
| 2026-09-20 | v1.110 | E2 验收：v11 + lane$\ge$4T 搜索为本工作的参照模型与搜索（18 个新层，17/18 实测最优，较生产 quick 快 13.2%，非 2T 计划 regret@1 中位 0%/最大 7.9%）；2T lane 按冻结规则移出搜索空间并记为限制（实测/预测 1.237，符号一致 0/18）。新增快速 planner `hot_wide_planner.py`（宽 lane 承载热点 + 4T 主体、按宽度负载系数、热点先行升序），模型目标下与 75 s LNS 相差 $-0.3\%$（中位）。默认消费方与 native 路径仍为 v8/v9。 |
| 2026-09-20 | v1.111 | E3 验收（18 个新层）：参照搜索（v11 + lane$\ge$4T LNS 90 s）较生产 quick 快 12.87%（中位）、17/18 实测最优，10 s 搜索在其 0.61% 以内；快速 planner 快 9.63%（18/18）但落后参照 3.82%。三者实测/预测 1.094--1.100，说明差距为搜索深度而非模型误差；模板事件评分 0.6 个点（272 ms）、顺序模式 0.8 个点（98 ms）、负载再平衡 0 个点，其余为 lane 间相位错开。 |
| 2026-09-20 | v1.112 | 生产接入：V3 window 表按 `machine_ids` 注册（该机器 lowering 默认带 window，实测 quick 快 1.61%）、`PlannedMoE` 增加 `hot_wide` 搜索模式、`MoePlannerRuntime` 增加 `event_calibration`/`search_mode` 与 `enable_moe_planner_fast`、`ProbeEventModel` 实现 T_iso 磁盘缓存接口。快速 planner 每层规划 1.85 ms（Python 预热）。native planner 路径未改。 |
| 2026-09-20 | v1.113 | 候选 v12/v13：P7 测得背景 lane 宽度对争用的影响（2T 背景最重、16T 最轻）并作为超额因子；P8 在真实背景下检验组合，定位 2T 低估 18--49% 的两项成因，v13 令 2T lane 在所有 phase 计为装载，复现 P8 的 2T 格（中位 0.978）。E2 上 2T 计划 1.237→1.144、含 2T 的 regret 中位 6.07%→0。验收在 E5（新层）。 |
| 2026-09-20 | v1.114 | E5 验收：v13 未过 W1/W2/W3（2T 计划实测/预测 1.281、regret 中位 11.95% 对 v11 的 4.45%、符号一致 0/18），按冻结规则保留 v11 为参照模型、2T 仍在搜索空间外。E5 第三次独立确认：$\ge$4T 搜索较生产 quick 快 9.97%、快速 planner 快 8.01%（均 18/18），两者与实测最优差 0.05% 与 3.03%。 |
| 2026-09-20 | v1.115 | 标定域约束：模型给出 `calibrated_widths`(2--32T) 与 `reliable_widths`(扣除 2T)，tail-pool 候选与 LNS 默认宽度按可信集过滤。依据 E8：1T 池化任务实测为预测的 3.25--4.25 倍，使 tail-pool 计划慢 1.0--1.3 ms 而打分认为快 0.7--3.3%。新增路由切片移动(默认关闭)：54 个层上最热 expert/工作量下界比值中位 0.38、最大 0.44，切片无收益。 |
| 2026-09-20 | v1.116 | Stage window 映射改为 thread-major：N domain 先按 worker 切连续 stripe（§1.1 的 $\sigma_s(j)$），窗口在 stripe 内部切分。取整恒等式给出 $R_s$、每 pass 的 $g_s$、每 worker 单窗口 footprint 与 A 扫描次数均不变，改变的只是列归属；C++ 与 `full_stage_geometry.py` 同步。证据：原型网格（加载 44 格中位 −0.033%、29 负；小 M 33 格中位 −0.574%、27 负；checksum 全等）与 E9 整计划 A/B（同一 checkout 的两个构建，6 个真实层 × 11 计划 = 66 点，A B B A 四 session）：中位 −0.027%、符号 37/66、p10/p90 −0.245%/+0.291%，构建内 session 差中位 0.261%；每层 Spearman 0.82--0.99、最快计划均未变；两构建 66 个计划输出 sha256 全等。V3 窗口表与 v11 window 项仍是在旧顺序下标定的，其逐 (M, width) 最优尚未重测。报告：`optimizations/fused_moe_sve/results/window_order_thread_major_20260920.md`。 |

Change record (2026-09-14, Lab): implemented predeclared matched block history/pressure interpolation and training-only extraction; zero/pooled-history controls, bounded domain, signed-delta and conditional-pressure limitations recorded. No active planner or production equation replacement.
