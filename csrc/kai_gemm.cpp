/*
 * KleidiAI GEMM 后端实现
 *
 * 基于 ARM KleidiAI 库的 NEON BFMMLA 微内核实现多线程分块 GEMM，
 * 支持 FP32 和 BF16 两种输出精度。本文件当前阶段仅实现权重预打包
 * 接口 ``kai_gemm_prepare``，其余接口会在后续任务中补齐。
 *
 * 编译条件：仅在 AArch64 平台且 KleidiAI 源码可用（定义了
 * ``FUSED_CPP_HAS_KLEIDIAI`` 宏）时启用真实实现，否则提供 stub。
 */

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "kai_gemm_config.h"

#if defined(__aarch64__) && defined(FUSED_CPP_HAS_KLEIDIAI)

#include <c10/util/BFloat16.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

extern "C" {
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/\
kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"
}

namespace fused_cpp {
namespace kai_gemm {

// 将 BF16 权重按位扩展为 FP32（Bf16 -> Fp32：把 16 位放到高半字）。
// KleidiAI 的 RHS pack 函数只接受 FP32 输入，因此 BF16 权重需要先转换。
static at::Tensor bf16_to_fp32(const at::Tensor& weight_bf16) {
    return weight_bf16.to(at::kFloat).contiguous();
}

// ---------------------------------------------------------------------------
// CPU affinity 辅助函数（仅 Linux）。
//
// bind_current_thread_to_cpu: 把当前线程绑定到单个 CPU。非 Linux 平台
// 为 no-op，返回 true 表示"本平台无需绑核，不视为失败"。
// ---------------------------------------------------------------------------
static bool bind_current_thread_to_cpu(int cpu_id) {
#if defined(__linux__)
    if (cpu_id < 0) return true;  // <0 表示不绑核
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    return pthread_setaffinity_np(
               pthread_self(), sizeof(mask), &mask) == 0;
#else
    (void)cpu_id;
    return true;
#endif
}

// ---------------------------------------------------------------------------
// 简易线程池：parallel_for(num_tasks, fn(thread_id, task_id))
//
// 设计目标：
//   - 固定大小 worker 线程；生命周期由上层（create/destroy_kai_thread_pool）
//     显式管理，跨多个 GEMM handler 可共享。
//   - 不依赖 OpenMP；worker 启动时调用 ``at::set_num_threads(1)``，防止
//     与 PyTorch intra-op 并行互相 oversubscribe。
//   - 只提供阻塞式 ``parallel_for``：当前线程也参与计算（作为 thread_id=0），
//     所以实际并发度 = 1（主线程）+ num_workers。
//   - 可选支持按 CPU ID 列表绑核：cpu_ids[0] 给调用线程（主线程），
//     cpu_ids[1..] 给 worker 线程。cpu_ids 为空时不绑核。
//   - 线程安全：``parallel_for`` 由 ``submit_mu_`` 互斥保护，允许多个
//     caller 线程同时持有同一个 pool 句柄而不会踩到任务队列。
//   - 提供 per-thread scratch buffer（``EnsureScratch`` / ``LhsScratch``
//     / ``Fp32TileScratch``），用于避免每个 handler 各自持有一份 buffer
//     导致的内存浪费；scratch 按 pool 的线程数天然分片，按最大需求扩容。
// ---------------------------------------------------------------------------
class KAIThreadPool {
 public:
    // num_workers: 额外工作线程数。实际并发度 = num_workers + 1（含调用线程）。
    // cpu_ids: 长度 0 或 num_workers + 1。若非空，cpu_ids[i+1] 作为
    //          worker i 的绑定核；cpu_ids[0] 用于 parallel_for 时临时
    //          把调用线程绑到那个核。
    KAIThreadPool(int num_workers, std::vector<int> cpu_ids)
        : num_workers_(num_workers), cpu_ids_(std::move(cpu_ids)) {
        if (!cpu_ids_.empty() &&
            static_cast<int>(cpu_ids_.size()) != num_workers_ + 1) {
            throw std::runtime_error(
                "KAIThreadPool: cpu_ids 长度必须为 0 或 num_workers+1");
        }
        const std::size_t num_threads =
            static_cast<std::size_t>(num_workers_) + 1;
        lhs_scratch_.resize(num_threads);
        fp32_scratch_.resize(num_threads);
        workers_.reserve(num_workers_);
        for (int i = 0; i < num_workers_; ++i) {
            const int worker_cpu = cpu_ids_.empty() ? -1 : cpu_ids_[i + 1];
            workers_.emplace_back([this, i, worker_cpu]() {
                this->worker_loop(i + 1, worker_cpu);
            });
        }
    }

    explicit KAIThreadPool(int num_workers)
        : KAIThreadPool(num_workers, std::vector<int>{}) {}

    ~KAIThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
            cv_.notify_all();
        }
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    KAIThreadPool(const KAIThreadPool&) = delete;
    KAIThreadPool& operator=(const KAIThreadPool&) = delete;

    // 并行执行 num_tasks 个任务；fn 的签名为 fn(thread_id, task_id)。
    // thread_id 范围：[0, num_workers_ + 1)。
    // 阻塞直到全部任务完成。
    // 使用 ``submit_mu_`` 保证同一时间只有一个 caller 在提交任务，
    // 不同 caller 串行复用同一 pool。
    void parallel_for(int num_tasks,
                      const std::function<void(int, int)>& fn) {
        if (num_tasks <= 0) return;
        std::lock_guard<std::mutex> submit_lk(submit_mu_);

        // 若配置了 cpu_ids，把当前调用线程（主线程）绑到第 0 个核。
        // 幂等操作：每次 parallel_for 都设一次，避免调用线程在别处被迁走。
        if (!cpu_ids_.empty()) {
            bind_current_thread_to_cpu(cpu_ids_[0]);
        }

        if (num_workers_ == 0) {
            // 单线程路径：全部任务在调用线程上顺序执行。
            for (int i = 0; i < num_tasks; ++i) fn(0, i);
            return;
        }

        {
            std::unique_lock<std::mutex> lk(mu_);
            current_fn_ = &fn;
            total_tasks_ = num_tasks;
            next_task_.store(0, std::memory_order_relaxed);
            done_count_.store(0, std::memory_order_relaxed);
            job_epoch_++;
            cv_.notify_all();
        }

        // 主线程（thread_id=0）也来拉取任务，直到任务全部完成。
        run_until_drained(0);

        // 等待所有 worker 也完成当前 job。
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this]() {
            return done_count_.load(std::memory_order_acquire) ==
                   num_workers_;
        });
        current_fn_ = nullptr;
    }

    int num_threads() const { return num_workers_ + 1; }

    // 确保 per-thread scratch 至少满足给定大小（按最大需求扩容）。
    // 必须在 parallel_for 外部或互斥保护下调用（当前由 run_gemm_impl
    // 在 parallel_for 前调用一次，保证串行扩容）。
    void EnsureScratch(std::size_t lhs_bytes, std::size_t fp32_elems) {
        for (auto& buf : lhs_scratch_) {
            if (buf.size() < lhs_bytes) buf.resize(lhs_bytes);
        }
        for (auto& buf : fp32_scratch_) {
            if (buf.size() < fp32_elems) buf.resize(fp32_elems);
        }
    }

    // 取 thread_id 对应的 LHS packed scratch 指针。
    std::uint8_t* LhsScratch(int thread_id) {
        return lhs_scratch_[static_cast<std::size_t>(thread_id)].data();
    }

    // 取 thread_id 对应的 FP32 tile scratch 指针（仅 BF16 输出路径使用）。
    float* Fp32TileScratch(int thread_id) {
        return fp32_scratch_[static_cast<std::size_t>(thread_id)].data();
    }

 private:
    void worker_loop(int thread_id, int cpu_id) {
        // 启动后优先绑核（若指定）。绑核失败不阻止继续执行，只是
        // 退化为无 affinity 的线程。
        bind_current_thread_to_cpu(cpu_id);
        // 防止 PyTorch intra-op 并行在 worker 内再次展开。
        at::set_num_threads(1);
        std::uint64_t last_epoch = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this, &last_epoch]() {
                    return stop_ || job_epoch_ != last_epoch;
                });
                if (stop_) return;
                last_epoch = job_epoch_;
            }
            run_until_drained(thread_id);
            if (done_count_.fetch_add(1, std::memory_order_acq_rel) + 1 ==
                num_workers_) {
                std::unique_lock<std::mutex> lk(mu_);
                done_cv_.notify_all();
            }
        }
    }

    // 以原子计数器拉取剩余任务并执行，直到任务队列被清空。
    void run_until_drained(int thread_id) {
        const auto& fn = *current_fn_;
        while (true) {
            int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_tasks_) break;
            fn(thread_id, idx);
        }
    }

    const int num_workers_;
    const std::vector<int> cpu_ids_;
    std::vector<std::thread> workers_;

    // 保证每次 parallel_for 会话独占 pool，支持多 caller 安全串行复用。
    std::mutex submit_mu_;

    std::mutex mu_;
    std::condition_variable cv_;       // 唤醒 worker 开始新 job
    std::condition_variable done_cv_;  // 唤醒主线程：job 完成
    bool stop_ = false;
    std::uint64_t job_epoch_ = 0;
    const std::function<void(int, int)>* current_fn_ = nullptr;
    int total_tasks_ = 0;
    std::atomic<int> next_task_{0};
    std::atomic<int> done_count_{0};

    // 每线程 scratch：个数 = num_workers_ + 1（含主线程 tid=0）。
    // 大小按最大需求扩容，避免跨 handler 反复分配。
    std::vector<std::vector<std::uint8_t>> lhs_scratch_;
    std::vector<std::vector<float>> fp32_scratch_;
};

// ---------------------------------------------------------------------------
// KAIGEMMHandler: GEMM 运行时上下文（纯数据容器）
//
// 成员：
//   - packed_weight_tensor: 保活用户传入的 packed RHS 张量（零拷贝引用）。
//   - packed_weight_ptr   : packed_weight_tensor.data_ptr() 的缓存。
//   - K, N                : 原始权重维度。
//   - Mc, Nc, Kc          : 分块参数（从编译期常量拷贝到 handler，便于将来按
//                           运行时 K/N 裁剪）。
//
// 说明：
//   handler 不再持有线程池、per-thread scratch 等资源。线程池和
//   scratch 生命周期由上层显式的 KAIThreadPool 管理，调用 kai_gemm
//   时显式传入。这使得同一个 handler 可以反复用于不同 pool，也可以被
//   多个 handler 共享同一个 pool（节省线程资源）。
// ---------------------------------------------------------------------------
struct KAIGEMMHandler {
    at::Tensor packed_weight_tensor;
    const void* packed_weight_ptr = nullptr;
    int64_t K = 0;
    int64_t N = 0;

    // 分块参数。
    std::size_t Mc = kMc;
    std::size_t Nc = kNc;
    std::size_t Kc = kKc;
};
}  // namespace kai_gemm
}  // namespace fused_cpp

// ---------------------------------------------------------------------------
// kai_gemm_prepare: 对权重 B 做离线预打包
//
//   weight : [K, N] 行主序 FP32 或 BF16 张量
//   bias   : 可选 [N] FP32 张量（用户不提供时传 None）
//   返回   : 一维 uint8 张量，包含 KleidiAI packed RHS 数据
// ---------------------------------------------------------------------------
at::Tensor kai_gemm_prepare(at::Tensor weight,
                            c10::optional<at::Tensor> bias) {
    TORCH_CHECK(weight.dim() == 2,
                "KAI GEMM prepare: weight 必须为 2D [K, N]，当前 dim=",
                weight.dim());
    TORCH_CHECK(weight.scalar_type() == at::kFloat ||
                    weight.scalar_type() == at::kBFloat16,
                "KAI GEMM prepare: weight 仅支持 float32 / bfloat16");
    TORCH_CHECK(weight.is_contiguous(),
                "KAI GEMM prepare: weight 必须 contiguous（行主序）");

    const int64_t K = weight.size(0);
    const int64_t N = weight.size(1);
    TORCH_CHECK(K > 0 && N > 0,
                "KAI GEMM prepare: K、N 必须为正，K=", K, " N=", N);

    // KleidiAI pack 函数只接受 FP32，BF16 权重先提升为 FP32。
    at::Tensor weight_f32;
    if (weight.scalar_type() == at::kBFloat16) {
        weight_f32 = fused_cpp::kai_gemm::bf16_to_fp32(weight);
    } else {
        weight_f32 = weight.contiguous();
    }

    // 处理可选 bias：必须为 [N] FP32。
    at::Tensor bias_f32;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        const at::Tensor& b = bias.value();
        TORCH_CHECK(b.dim() == 1,
                    "KAI GEMM prepare: bias 必须为 1D [N]，当前 dim=",
                    b.dim());
        TORCH_CHECK(b.size(0) == N,
                    "KAI GEMM prepare: bias 长度必须等于 N=", N,
                    "，实际 ", b.size(0));
        TORCH_CHECK(b.scalar_type() == at::kFloat,
                    "KAI GEMM prepare: bias dtype 必须为 float32");
        bias_f32 = b.contiguous();
        bias_ptr = bias_f32.data_ptr<float>();
    }

    const std::size_t nr = fused_cpp::kai_gemm::kNr;
    const std::size_t kr = fused_cpp::kai_gemm::kKr;
    const std::size_t sr = 1;  // KleidiAI 当前要求 sr == 1

    const std::size_t packed_size =
        kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
            static_cast<std::size_t>(N), static_cast<std::size_t>(K),
            nr, kr);
    TORCH_CHECK(packed_size > 0,
                "KAI GEMM prepare: packed_size 计算失败（为 0）");

    // 分配一维 uint8 张量用于承载 packed RHS。
    at::Tensor packed = at::empty(
        {static_cast<int64_t>(packed_size)},
        at::TensorOptions().dtype(at::kByte));

    // rhs_stride: 行主序 [K, N]，每行 N 个 fp32 元素。
    const std::size_t rhs_stride =
        static_cast<std::size_t>(N) * sizeof(float);

    kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
        /*num_groups=*/1,
        static_cast<std::size_t>(N),
        static_cast<std::size_t>(K),
        nr, kr, sr,
        rhs_stride,
        weight_f32.data_ptr<float>(),
        bias_ptr,      // 可能为 nullptr；KleidiAI 内部会用 0 填充 bias 区
        /*scale=*/nullptr,
        packed.data_ptr(),
        /*extra_bytes=*/0,
        /*params=*/nullptr);

    return packed;
}

namespace fused_cpp {
namespace kai_gemm {

// ---------------------------------------------------------------------------
// PoolRegistry: 注册所有存活的 KAIThreadPool handle。
//
// 目的：
//   1) 提供 int64_t <-> KAIThreadPool* 的合法性校验；
//   2) 进程退出时通过 atexit 兜底 join 残留 worker 线程，避免
//      "terminate called without an active exception"。
//
// 关键设计（规避 C++ 静态析构与 atexit 栈交叠导致的 use-after-free）：
//   - Registry 采用 "leaky singleton" 惯用法：实例由 ``new`` 一次性分配、
//     永不 ``delete``。进程退出时内存由 OS 回收。这样 C++ runtime 不会
//     在静态析构阶段触碰 ``pools_``。
//   - atexit 回调只做 ``ShutdownAll``：取出 pools 并在栈对象上析构；
//     之后将 ``shutdown_done_`` 置位，后续任何 Register/Unregister/Get
//     都变成安全 no-op，以防 Python/PyTorch 的 late finalizer 再次触达。
//   - 这一设计避免了 "atexit lambda 访问已析构的 Meyers singleton" 的
//     double-free 陷阱（观测到的现象：bench 结束、CSV 写完后进程退出阶段
//     出现 "double free or corruption (!prev)"）。
//
// 并发：pool 的创建/销毁是低频事件，用一把全局互斥锁保护注册表即可。
// ---------------------------------------------------------------------------
class PoolRegistry {
 public:
    static PoolRegistry& Instance() {
        // leaky singleton：new 出来后永不 delete。
        // kInstance 本身是 trivially-destructible 指针，不会产生静态析构。
        static PoolRegistry* const kInstance = CreateInstance();
        return *kInstance;
    }

    // 注册新创建的 pool，所有权转移到注册表。
    // 返回裸指针（int64_t），作为 Python 侧的 handle。
    // 若 shutdown 已触发（进程退出兜底期），则拒绝注册并立即析构传入 pool。
    int64_t Register(std::unique_ptr<KAIThreadPool> pool) {
        KAIThreadPool* raw = pool.get();
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (shutdown_done_) return 0;
            pools_[reinterpret_cast<int64_t>(raw)] = std::move(pool);
        }
        return reinterpret_cast<int64_t>(raw);
    }

    // 从注册表移除并销毁 pool。句柄非法（已销毁或从未注册）时返回 false。
    bool Unregister(int64_t handle) {
        std::unique_ptr<KAIThreadPool> to_destroy;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (shutdown_done_) return false;
            auto it = pools_.find(handle);
            if (it == pools_.end()) return false;
            to_destroy = std::move(it->second);
            pools_.erase(it);
        }
        // to_destroy 在锁外析构，避免 worker join 时与其它锁相互死锁。
        return true;
    }

    // 按句柄取指针；非法或 shutdown 之后返回 nullptr。用于 kai_gemm 入参校验。
    KAIThreadPool* Get(int64_t handle) const {
        if (handle == 0) return nullptr;
        std::lock_guard<std::mutex> lk(mu_);
        if (shutdown_done_) return nullptr;
        auto it = pools_.find(handle);
        return it == pools_.end() ? nullptr : it->second.get();
    }

    // 进程退出兜底：释放所有还没销毁的 pool。
    // 语义：
    //   - 把 pools_ move 到栈上，锁外析构（触发 worker join）；
    //   - 置 shutdown_done_ = true，使后续 Register/Unregister/Get 成 no-op，
    //     防止 Python/PyTorch 的 late finalizer 在此之后再次访问。
    void ShutdownAll() {
        std::unordered_map<int64_t, std::unique_ptr<KAIThreadPool>> moved;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (shutdown_done_) return;
            moved = std::move(pools_);
            pools_.clear();
            shutdown_done_ = true;
        }
        // 在锁外统一析构，顺序由 map 内部决定。
        moved.clear();
    }

 private:
    static PoolRegistry* CreateInstance() {
        auto* inst = new PoolRegistry();
        // 注册 atexit 兜底；进程退出时统一清理残留 pool。
        // 注意：这里不再访问 Instance()，而是直接捕获裸指针；既避免
        // 再次命中 Meyers singleton 的构造/析构路径，也保证即使 C++
        // runtime 已进入静态析构阶段，裸指针依然有效（leaky）。
        std::atexit([]() {
            // 通过 Instance() 拿到同一个 leaky 对象；inst 已被捕获时
            // 可省去一次 static 初始化检查，但这里为简洁仍走 Instance()。
            PoolRegistry::Instance().ShutdownAll();
        });
        return inst;
    }

    PoolRegistry() = default;
    // Leaky singleton：实例永不 delete，析构理论上不会被调用；
    // 置为 private 以禁止外部析构，但保留默认析构体以允许 ``new`` 表达式
    // 处理构造期异常路径（C++17 要求 new 表达式可访问析构函数）。
    ~PoolRegistry() = default;
    PoolRegistry(const PoolRegistry&) = delete;
    PoolRegistry& operator=(const PoolRegistry&) = delete;

    mutable std::mutex mu_;
    std::unordered_map<int64_t, std::unique_ptr<KAIThreadPool>> pools_;
    bool shutdown_done_ = false;
};

}  // namespace kai_gemm
}  // namespace fused_cpp

// ---------------------------------------------------------------------------
// create_kai_thread_pool: 创建一个独立线程池资源
//
//   cpu_ids: 绑核 CPU 列表；长度即总并发度（含调用线程）。
//            cpu_ids[0] 用于 parallel_for 时临时绑主调用线程，
//            cpu_ids[1..] 由 worker 线程启动时各自绑定。
//            若 cpu_ids 为空，则退化为 1 线程、不绑核（仅主线程）。
//   返回   : int64_t pool handle；0 保留为 "无效/不使用 pool" 语义。
//
// 线程池的生命周期完全由调用者持有：需要时 destroy_kai_thread_pool 显式释放，
// 或由进程退出时的 atexit 兜底清理。
// ---------------------------------------------------------------------------
int64_t create_kai_thread_pool(std::vector<int64_t> cpu_ids) {
    std::vector<int> cpus;
    cpus.reserve(cpu_ids.size());
    for (int64_t c : cpu_ids) {
        TORCH_CHECK(c >= 0,
                    "KAIThreadPool: cpu_ids 元素必须 >= 0，实际 ", c);
        cpus.push_back(static_cast<int>(c));
    }

    std::unique_ptr<fused_cpp::kai_gemm::KAIThreadPool> pool;
    if (cpus.empty()) {
        // 空列表：1 线程、不绑核（num_workers=0）。
        pool = std::make_unique<fused_cpp::kai_gemm::KAIThreadPool>(0);
    } else {
        const int num_workers = static_cast<int>(cpus.size()) - 1;
        pool = std::make_unique<fused_cpp::kai_gemm::KAIThreadPool>(
            num_workers, std::move(cpus));
    }
    return fused_cpp::kai_gemm::PoolRegistry::Instance().Register(
        std::move(pool));
}

// ---------------------------------------------------------------------------
// destroy_kai_thread_pool: 销毁线程池资源
//
//   pool_handle: create_kai_thread_pool 返回的 handle；0 或非法句柄安全忽略。
// ---------------------------------------------------------------------------
void destroy_kai_thread_pool(int64_t pool_handle) {
    if (pool_handle == 0) return;
    fused_cpp::kai_gemm::PoolRegistry::Instance().Unregister(pool_handle);
}

// ---------------------------------------------------------------------------
// create_kai_gemm_handler: 创建纯数据 handler
//
//   packed_weight : 由 kai_gemm_prepare 返回的 1D uint8 张量
//   K, N          : 原始权重维度（由 Python 侧传入）
//
// handler 本身不持有任何线程资源；执行时需要通过 kai_gemm 的 pool_handle
// 参数显式传入线程池。
// ---------------------------------------------------------------------------
int64_t create_kai_gemm_handler(at::Tensor packed_weight,
                                int64_t K, int64_t N) {
    TORCH_CHECK(packed_weight.dim() == 1,
                "KAI GEMM handler: packed_weight 必须为 1D uint8 张量");
    TORCH_CHECK(packed_weight.scalar_type() == at::kByte,
                "KAI GEMM handler: packed_weight dtype 必须为 uint8");
    TORCH_CHECK(packed_weight.is_contiguous(),
                "KAI GEMM handler: packed_weight 必须 contiguous");
    TORCH_CHECK(K > 0 && N > 0,
                "KAI GEMM handler: K、N 必须为正，K=", K, " N=", N);

    // 校验 packed_weight 大小与 (K, N) 匹配。
    const std::size_t expected =
        kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
            static_cast<std::size_t>(N), static_cast<std::size_t>(K),
            fused_cpp::kai_gemm::kNr, fused_cpp::kai_gemm::kKr);
    TORCH_CHECK(static_cast<std::size_t>(packed_weight.numel()) == expected,
                "KAI GEMM handler: packed_weight 大小与 (K, N) 不匹配，"
                "期望 ", expected, " bytes，实际 ",
                packed_weight.numel(), " bytes");

    auto* handler = new fused_cpp::kai_gemm::KAIGEMMHandler();
    handler->packed_weight_tensor = packed_weight;  // 保活
    handler->packed_weight_ptr = packed_weight.data_ptr();
    handler->K = K;
    handler->N = N;
    handler->Mc = fused_cpp::kai_gemm::kMc;
    handler->Nc = fused_cpp::kai_gemm::kNc;
    handler->Kc = fused_cpp::kai_gemm::kKc;
    return reinterpret_cast<int64_t>(handler);
}

// ---------------------------------------------------------------------------
// release_kai_gemm_handler: 释放 handler
// ---------------------------------------------------------------------------
void release_kai_gemm_handler(int64_t handler_ptr) {
    auto* handler =
        reinterpret_cast<fused_cpp::kai_gemm::KAIGEMMHandler*>(handler_ptr);
    if (handler != nullptr) {
        delete handler;  // 仅解除 packed_weight_tensor 引用。
    }
}

namespace fused_cpp {
namespace kai_gemm {

// ---------------------------------------------------------------------------
// 对一个 Mc 子块执行完整 K、当前 Nc 片的计算：
//   1. 打包该 Mc 子块的 LHS（完整 K 维度）到 thread-local buffer。
//   2. 按 nr 面板逐列调用 KleidiAI FP32 微内核。
//
// 说明：KleidiAI 微内核一次调用内部会完整遍历 K，不支持外层对 K 累加。
// 因此我们在 Nc、Mc 两层做分块（使 B 分片驻留 L3、A 分块驻留 L2），
// 内层 nr 面板循环，微内核内部自行按 kMr/kKr 对 mr×nr tile 迭代。
//
// BF16 输出路径采用"微内核级就地转换"：每处理完一个 nr 面板，把 FP32
// 结果从 thread-local tile buffer 立即转换为 BF16 写回用户 output。
// 这样 FP32 中间结果只占 Mc*kNr（约数 KB，驻留 L1），规避了整块
// Mc*Nc 的 scratch 开销和重遍历。
//
//   dst_f32_base: 非空时，微内核把每个 nr 面板直接写入该 FP32 输出；
//                 与之对应 bf16_dst_base 必须为 nullptr。
//   bf16_dst_base:非空时，微内核写入 thread-local ``fp32_tile``，
//                 立即逐行转换为 BF16 写入 bf16_dst_base 对应列位置。
//
// 两个输出参数互斥：恰好一个为非空。
// ---------------------------------------------------------------------------
static void gemm_mc_block(
    const float* lhs_base,            // 指向 A 开头 [m0, 0] 的 FP32 指针
    int64_t lhs_stride_bytes,
    const std::uint8_t* rhs_packed,   // 指向当前 Nc 切片的 packed RHS
    float* dst_f32_base,              // FP32 输出起点（或 nullptr）
    std::size_t dst_f32_stride_elems, // FP32 输出的行 stride（元素数）
    c10::BFloat16* bf16_dst_base,     // BF16 输出起点（或 nullptr）
    std::size_t bf16_dst_stride_elems,// BF16 输出的行 stride（元素数）
    std::size_t mc_rows,
    std::size_t nc_cols,
    std::size_t K,
    float clamp_min, float clamp_max,
    std::uint8_t* lhs_packed_buffer,
    float* fp32_tile /* 大小 >= mc_rows*kNr，仅 BF16 路径使用 */) {
    // 1) pack LHS 的 [mc_rows, K] 子块到 thread-local buffer（完整 K 维度）。
    kai_run_lhs_quant_pack_bf16p8x4_f32_neon(
        mc_rows, K, kMr, kKr, /*sr=*/1, /*m_idx_start=*/0,
        lhs_base,
        static_cast<std::size_t>(lhs_stride_bytes),
        lhs_packed_buffer);

    // 2) 内层按 nr 列遍历（微内核内部会再按 mr 迭代）。
    for (std::size_t j = 0; j < nc_cols; j += kNr) {
        const std::size_t n_block = std::min<std::size_t>(kNr, nc_cols - j);

        // packed RHS 子片 = rhs_packed + 列偏移。
        const std::size_t rhs_off =
            kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
                j, K, kNr, kKr);
        const std::uint8_t* rhs_sub = rhs_packed + rhs_off;

        if (dst_f32_base != nullptr) {
            // ---- FP32 输出：微内核直接写用户 output ----
            float* dst_sub = dst_f32_base + j;
            const std::size_t dst_row_bytes =
                dst_f32_stride_elems * sizeof(float);
            kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                mc_rows, n_block, K,
                lhs_packed_buffer, rhs_sub,
                dst_sub, dst_row_bytes, sizeof(float),
                clamp_min, clamp_max);
        } else {
            // ---- BF16 输出：微内核先写 FP32 tile，再转换到 output ----
            // tile 行 stride = kNr（紧凑行主序存放 mc_rows × kNr 的 FP32）。
            const std::size_t tile_row_bytes = kNr * sizeof(float);
            kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                mc_rows, n_block, K,
                lhs_packed_buffer, rhs_sub,
                fp32_tile, tile_row_bytes, sizeof(float),
                clamp_min, clamp_max);

            // 把 tile buffer 中 mc_rows × n_block 的 FP32 数据转换为 BF16
            // 写到 bf16_dst_base + 列偏移 j。FP32 结果仍然在 L1 中，
            // 转换过程顺序访问，cache 表现良好。
            c10::BFloat16* dst_col_base = bf16_dst_base + j;
            for (std::size_t i = 0; i < mc_rows; ++i) {
                const float* src_row = fp32_tile + i * kNr;
                c10::BFloat16* dst_row =
                    dst_col_base + i * bf16_dst_stride_elems;
                for (std::size_t jj = 0; jj < n_block; ++jj) {
                    dst_row[jj] = static_cast<c10::BFloat16>(src_row[jj]);
                }
            }
        }
    }
}

// 运行 GEMM 主循环：
//   - FP32 输出：微内核直接写入用户 output 张量（零拷贝）。
//   - BF16 输出：微内核按 nr 面板写入 thread-local FP32 tile buffer，
//                每个面板完成后立即转换为 BF16 写入 output
//                （micro-kernel 级就地转换，FP32 中间结果驻留 L1）。
//
// pool == nullptr 或 pool->num_threads() == 1 时走单线程路径；否则通过
// pool->parallel_for 并行执行 Mc / Mr 粒度的任务。per-thread scratch
// buffer 由 pool 按需扩容（EnsureScratch），避免 handler 各自持有一份。
static void run_gemm_impl(
    KAIGEMMHandler& handler,
    KAIThreadPool* pool,
    const float* lhs_ptr, int64_t lhs_stride_bytes,
    void* dst_ptr, int64_t dst_stride_row_bytes,
    bool dst_is_bf16,
    std::size_t M, std::size_t N, std::size_t K) {
    const auto* rhs_packed =
        static_cast<const std::uint8_t*>(handler.packed_weight_ptr);
    constexpr float kClampMin = -FLT_MAX;
    constexpr float kClampMax = FLT_MAX;

    const std::size_t Nc = handler.Nc;
    const std::size_t Mc = handler.Mc;

    // 计算该 handler 需要的 per-thread scratch 大小并让 pool 就地扩容。
    // 注意：多线程路径下 pool 必须非空；scratch 分配前置于 parallel_for
    // 之外，保证 worker 拿到的 buffer 已经就绪且足够大。
    const std::size_t lhs_buf_bytes =
        kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon(
            Mc, K, kMr, kKr, /*sr=*/1);
    const std::size_t tile_elems = dst_is_bf16 ? (Mc * kNr) : 0;

    // 单线程路径：栈上分配一次性 scratch，避免依赖 pool。
    // （用 std::vector 保证对齐和释放；Mc*K 通常在几十 KB，开销可忽略。）
    std::vector<std::uint8_t> local_lhs;
    std::vector<float> local_tile;
    const bool use_pool = (pool != nullptr && pool->num_threads() > 1);

    if (use_pool) {
        pool->EnsureScratch(lhs_buf_bytes, tile_elems);
    } else {
        local_lhs.resize(lhs_buf_bytes);
        if (tile_elems > 0) local_tile.resize(tile_elems);
    }

    // 获取 tid 对应的 scratch 指针。
    auto get_lhs_buf = [&](int tid) -> std::uint8_t* {
        return use_pool ? pool->LhsScratch(tid) : local_lhs.data();
    };
    auto get_tile_buf = [&](int tid) -> float* {
        return use_pool ? pool->Fp32TileScratch(tid) : local_tile.data();
    };

    // 为单个 Mc 块处理：给定当前 Nc 切片和 m0/mc。
    auto run_one_mc = [&](const std::uint8_t* rhs_n_slice,
                          std::size_t n0, std::size_t nc,
                          std::size_t m0, std::size_t mc,
                          int tid) {
        std::uint8_t* lhs_buf = get_lhs_buf(tid);
        const float* lhs_base =
            lhs_ptr + m0 * (lhs_stride_bytes / sizeof(float));

        if (dst_is_bf16) {
            float* tile = get_tile_buf(tid);
            auto* dst_bytes = static_cast<std::uint8_t*>(dst_ptr)
                              + m0 * dst_stride_row_bytes
                              + n0 * sizeof(c10::BFloat16);
            gemm_mc_block(
                lhs_base, lhs_stride_bytes,
                rhs_n_slice,
                /*dst_f32_base=*/nullptr, /*dst_f32_stride_elems=*/0,
                reinterpret_cast<c10::BFloat16*>(dst_bytes),
                static_cast<std::size_t>(dst_stride_row_bytes /
                                         sizeof(c10::BFloat16)),
                mc, nc, K, kClampMin, kClampMax, lhs_buf, tile);
        } else {
            auto* dst_bytes = static_cast<std::uint8_t*>(dst_ptr)
                              + m0 * dst_stride_row_bytes
                              + n0 * sizeof(float);
            gemm_mc_block(
                lhs_base, lhs_stride_bytes,
                rhs_n_slice,
                reinterpret_cast<float*>(dst_bytes),
                static_cast<std::size_t>(dst_stride_row_bytes /
                                         sizeof(float)),
                /*bf16_dst_base=*/nullptr, /*bf16_dst_stride_elems=*/0,
                mc, nc, K, kClampMin, kClampMax, lhs_buf,
                /*fp32_tile=*/nullptr);
        }
    };

    // Nc 外层循环。
    for (std::size_t n0 = 0; n0 < N; n0 += Nc) {
        const std::size_t nc = std::min(Nc, N - n0);

        // 当前 Nc 切片对应的 packed RHS 起始偏移。
        const std::size_t rhs_off_nc =
            kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
                n0, K, kNr, kKr);
        const std::uint8_t* rhs_n_slice = rhs_packed + rhs_off_nc;

        // Mc 循环：确定 Mc 块总数 num_mc。
        const std::size_t num_mc = (M + Mc - 1) / Mc;

        // 单线程路径：纯串行按 Mc 遍历。
        if (!use_pool) {
            for (std::size_t mi = 0; mi < num_mc; ++mi) {
                const std::size_t m0 = mi * Mc;
                const std::size_t mc = std::min(Mc, M - m0);
                run_one_mc(rhs_n_slice, n0, nc, m0, mc, /*tid=*/0);
            }
            continue;  // 处理下一个 Nc 片
        }

        // 多线程路径：自适应粒度。
        //   - 若 num_mc >= pool_threads：按 Mc 粒度切分，保 A 分块驻留 L2。
        //   - 否则：降级到 Mr 粒度切分，优先让更多线程拿到活。
        //
        // Mr 粒度时，每个任务只处理 Mr 行（最后一块可能不足 Mr），LHS
        // packed buffer（按 Mc*K 预分配）足以承载一次 Mr*K 的打包。
        const int pool_threads = pool->num_threads();
        if (num_mc >= static_cast<std::size_t>(pool_threads)) {
            // 够喂饱：按 Mc 派发。
            pool->parallel_for(
                static_cast<int>(num_mc),
                [&](int tid, int task_idx) {
                    const std::size_t m0 =
                        static_cast<std::size_t>(task_idx) * Mc;
                    const std::size_t mc = std::min(Mc, M - m0);
                    run_one_mc(rhs_n_slice, n0, nc, m0, mc, tid);
                });
        } else {
            // 不够：降级到 Mr 粒度派发。
            const std::size_t num_mr =
                (M + kMr - 1) / kMr;
            pool->parallel_for(
                static_cast<int>(num_mr),
                [&](int tid, int task_idx) {
                    const std::size_t m0 =
                        static_cast<std::size_t>(task_idx) * kMr;
                    const std::size_t mc = std::min(
                        static_cast<std::size_t>(kMr), M - m0);
                    run_one_mc(rhs_n_slice, n0, nc, m0, mc, tid);
                });
        }
    }
}

}  // namespace kai_gemm
}  // namespace fused_cpp

// ---------------------------------------------------------------------------
// kai_gemm: 执行 GEMM
//
//   output      : [M, N] 预分配的输出张量（FP32 或 BF16）
//   input       : [M, K] FP32 contiguous 输入张量
//   handler_ptr : create_kai_gemm_handler 返回的指针
//   pool_handle : create_kai_thread_pool 返回的 handle；0 表示单线程路径
// ---------------------------------------------------------------------------
void kai_gemm(at::Tensor output, at::Tensor input,
              int64_t handler_ptr, int64_t pool_handle) {
    auto* handler_raw =
        reinterpret_cast<fused_cpp::kai_gemm::KAIGEMMHandler*>(handler_ptr);
    TORCH_CHECK(handler_raw != nullptr, "KAI GEMM: handler 指针为空");
    auto& handler = *handler_raw;

    TORCH_CHECK(input.dim() == 2,
                "KAI GEMM: input 必须为 2D [M, K]，实际 dim=", input.dim());
    TORCH_CHECK(output.dim() == 2,
                "KAI GEMM: output 必须为 2D [M, N]，实际 dim=", output.dim());
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "KAI GEMM: input dtype 必须为 float32");
    TORCH_CHECK(input.is_contiguous(),
                "KAI GEMM: input 必须 contiguous");
    TORCH_CHECK(output.is_contiguous(),
                "KAI GEMM: output 必须 contiguous");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    const int64_t N = output.size(1);
    TORCH_CHECK(K == handler.K,
                "KAI GEMM: input K 与 handler 不匹配，期望 ",
                handler.K, "，实际 ", K);
    TORCH_CHECK(N == handler.N,
                "KAI GEMM: output N 与 handler 不匹配，期望 ",
                handler.N, "，实际 ", N);
    TORCH_CHECK(output.size(0) == M,
                "KAI GEMM: output M 与 input M 不匹配，期望 ",
                M, "，实际 ", output.size(0));

    if (M == 0) return;  // 空输入直接返回。

    // 解析 pool handle（0 表示单线程路径，走本地 scratch）。
    fused_cpp::kai_gemm::KAIThreadPool* pool = nullptr;
    if (pool_handle != 0) {
        pool = fused_cpp::kai_gemm::PoolRegistry::Instance().Get(pool_handle);
        TORCH_CHECK(pool != nullptr,
                    "KAI GEMM: pool_handle 非法或已销毁，handle=",
                    pool_handle);
    }

    const float* lhs_ptr = input.data_ptr<float>();
    const int64_t lhs_stride_bytes =
        static_cast<int64_t>(K) * static_cast<int64_t>(sizeof(float));
    void* dst_ptr = output.data_ptr();
    const int64_t dst_stride_row_bytes =
        static_cast<int64_t>(N) * static_cast<int64_t>(output.element_size());

    if (output.scalar_type() == at::kFloat) {
        fused_cpp::kai_gemm::run_gemm_impl(
            handler, pool, lhs_ptr, lhs_stride_bytes,
            dst_ptr, dst_stride_row_bytes,
            /*dst_is_bf16=*/false,
            static_cast<std::size_t>(M),
            static_cast<std::size_t>(N),
            static_cast<std::size_t>(K));
    } else if (output.scalar_type() == at::kBFloat16) {
        fused_cpp::kai_gemm::run_gemm_impl(
            handler, pool, lhs_ptr, lhs_stride_bytes,
            dst_ptr, dst_stride_row_bytes,
            /*dst_is_bf16=*/true,
            static_cast<std::size_t>(M),
            static_cast<std::size_t>(N),
            static_cast<std::size_t>(K));
    } else {
        TORCH_CHECK(false,
                    "KAI GEMM: output dtype 仅支持 float32 和 bfloat16");
    }
}

#else  // !(__aarch64__ && FUSED_CPP_HAS_KLEIDIAI) -- stub 实现

at::Tensor kai_gemm_prepare(at::Tensor /*weight*/,
                            c10::optional<at::Tensor> /*bias*/) {
    throw std::runtime_error(
        "KAI GEMM: KleidiAI 后端不可用（需要 AArch64 且构建时启用 KleidiAI）");
}

int64_t create_kai_thread_pool(std::vector<int64_t> /*cpu_ids*/) {
    throw std::runtime_error(
        "KAI GEMM: KleidiAI 后端不可用（需要 AArch64 且构建时启用 KleidiAI）");
}

void destroy_kai_thread_pool(int64_t /*pool_handle*/) {
    // stub 实现下不持有任何资源，直接忽略。
}

int64_t create_kai_gemm_handler(at::Tensor /*packed_weight*/,
                                int64_t /*K*/, int64_t /*N*/) {
    throw std::runtime_error(
        "KAI GEMM: KleidiAI 后端不可用（需要 AArch64 且构建时启用 KleidiAI）");
}

void release_kai_gemm_handler(int64_t /*handler_ptr*/) {
    // stub 实现下不持有任何资源，直接忽略。
}

void kai_gemm(at::Tensor /*output*/, at::Tensor /*input*/,
              int64_t /*handler_ptr*/, int64_t /*pool_handle*/) {
    throw std::runtime_error(
        "KAI GEMM: KleidiAI 后端不可用（需要 AArch64 且构建时启用 KleidiAI）");
}

#endif  // __aarch64__ && FUSED_CPP_HAS_KLEIDIAI
