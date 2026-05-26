/*
 * ACL GEMM 后端实现
 *
 * 基于 ARM Compute Library (ACL) 的 NEGEMM 实现矩阵乘法，
 * 支持权重预打包（prepare/pack）以提升推理性能。
 *
 * 编译条件：仅在 AArch64 平台且 ACL 可用时启用。
 */

#include <torch/extension.h>
#include <stdexcept>
#include <cstring>

#ifdef __aarch64__

#include "arm_compute/core/Types.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/function_info/GEMMInfo.h"

using namespace arm_compute;

// ACL GEMM Handler 结构体
// 持有 NEGEMM 实例和预打包的权重张量
struct ACLGEMMHandler {
    NEGEMM gemm;
    Tensor weight_tensor;  // 权重 B，prepare 后内部已打包
    Tensor input_tensor;   // 持久化输入张量（import_memory 方式复用）
    Tensor output_tensor;  // 持久化输出张量（import_memory 方式复用）
    int64_t K;             // 输入维度
    int64_t N;             // 输出维度
    int64_t cached_M;      // 上次 configure 时的 M 值，-1 表示未初始化
    DataType acl_dtype;    // ACL 数据类型
    bool fast_math;        // 是否启用低精度加速路径

    ~ACLGEMMHandler() {
        input_tensor.allocator()->free();
        output_tensor.allocator()->free();
        weight_tensor.allocator()->free();
    }
};

// 将 PyTorch dtype 映射到 ACL DataType
static DataType torch_dtype_to_acl(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return DataType::F32;
        case at::kBFloat16:
            return DataType::BFLOAT16;
        default:
            throw std::runtime_error(
                "ACL GEMM: 不支持的数据类型，仅支持 float32 和 bfloat16");
    }
}

// 创建 ACL GEMM handler 并预打包权重
// weight: [K, N] 的 PyTorch 张量
// num_threads: ACL Scheduler 线程数，0 表示使用默认值
// 返回 handler 指针（转为 int64_t）
int64_t create_acl_gemm_handler(at::Tensor weight, int64_t num_threads,
                                bool fast_math) {
    TORCH_CHECK(weight.dim() == 2,
                "ACL GEMM: 权重张量必须为 2D，当前维度: ", weight.dim());
    TORCH_CHECK(weight.scalar_type() == at::kFloat ||
                weight.scalar_type() == at::kBFloat16,
                "ACL GEMM: 仅支持 float32 和 bfloat16 数据类型");

    // 确保权重连续
    weight = weight.contiguous();

    int64_t K = weight.size(0);
    int64_t N = weight.size(1);
    DataType acl_dtype = torch_dtype_to_acl(weight.scalar_type());

    auto *handler = new ACLGEMMHandler();
    handler->K = K;
    handler->N = N;
    handler->acl_dtype = acl_dtype;
    handler->fast_math = fast_math;
    handler->cached_M = -1;  // 标记为未初始化

    // ACL TensorShape 使用列优先格式: TensorShape(cols, rows)
    // 对于 NEGEMM: C = A × B，其中 A[M, K], B[K, N], C[M, N]
    // ACL 中 A 的 shape 为 TensorShape(K, M)
    //         B 的 shape 为 TensorShape(N, K)
    //         C 的 shape 为 TensorShape(N, M)

    // 初始化权重张量 B[K, N] -> ACL TensorShape(N, K)
    handler->weight_tensor.allocator()->init(
        TensorInfo(TensorShape(static_cast<unsigned int>(N),
                               static_cast<unsigned int>(K)),
                   1, acl_dtype));

    // 使用 M=1 作为初始占位来 configure NEGEMM
    // 后续 run 时如果 M 变化会 reconfigure
    handler->input_tensor.allocator()->init(
        TensorInfo(TensorShape(static_cast<unsigned int>(K), 1u),
                   1, acl_dtype));
    handler->output_tensor.allocator()->init(
        TensorInfo(TensorShape(static_cast<unsigned int>(N), 1u),
                   1, acl_dtype));

    GEMMInfo gemm_info;
    gemm_info.set_fast_math(fast_math);

    // 先 validate，避免不支持的配置导致 segfault
    auto status = NEGEMM::validate(
        handler->input_tensor.info(), handler->weight_tensor.info(), nullptr,
        handler->output_tensor.info(), 1.0f, 0.0f, gemm_info);
    if (status.error_code() != ErrorCode::OK) {
        delete handler;
        TORCH_CHECK(false,
                    "ACL GEMM: 当前平台不支持该配置 (dtype=",
                    weight.scalar_type(), ", K=", K, ", N=", N,
                    "): ", status.error_description());
    }

    handler->gemm.configure(&handler->input_tensor, &handler->weight_tensor,
                            nullptr, &handler->output_tensor,
                            1.0f, 0.0f, gemm_info);

    // 分配权重内存并填充数据
    handler->weight_tensor.allocator()->allocate();

    // 将 PyTorch 权重数据拷贝到 ACL 张量
    auto *dst = handler->weight_tensor.buffer();
    auto *src = weight.data_ptr();
    std::memcpy(dst, src, weight.nbytes());

    // 调用 prepare() 对权重进行预打包（reorder/pack）
    // prepare 会将权重从原始布局转换为适合 SIMD 内核的交错布局，
    // 后续 reconfigure 时不会重新 pack 已 prepare 过的权重
    handler->gemm.prepare();

    return reinterpret_cast<int64_t>(handler);
}

// 执行 ACL GEMM: output = input × weight^T
// output: [M, N] 预分配的输出张量
// input:  [M, K] 输入张量（已 reshape 为 2D）
// bias:   [N] 可选偏置
// handler_ptr: handler 指针
void acl_gemm(at::Tensor output, at::Tensor input,
              c10::optional<at::Tensor> bias, int64_t handler_ptr) {
    auto *handler = reinterpret_cast<ACLGEMMHandler *>(handler_ptr);
    TORCH_CHECK(handler != nullptr, "ACL GEMM: handler 指针为空");

    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = handler->N;

    TORCH_CHECK(K == handler->K,
                "ACL GEMM: 输入 K 维度不匹配，期望 ", handler->K,
                "，实际 ", K);

    // 当 M 发生变化时，需要 reconfigure NEGEMM
    // reconfigure 只更新输入/输出的 shape 信息，不会重新 pack 已 prepare 过的权重
    if (M != handler->cached_M) {
        // 释放旧的 import_memory 引用
        handler->input_tensor.allocator()->free();
        handler->output_tensor.allocator()->free();

        // 重新初始化输入/输出张量的 TensorInfo
        handler->input_tensor.allocator()->init(
            TensorInfo(TensorShape(static_cast<unsigned int>(K),
                                   static_cast<unsigned int>(M)),
                       1, handler->acl_dtype));
        handler->output_tensor.allocator()->init(
            TensorInfo(TensorShape(static_cast<unsigned int>(N),
                                   static_cast<unsigned int>(M)),
                       1, handler->acl_dtype));

        GEMMInfo gemm_info;
        gemm_info.set_fast_math(handler->fast_math);

        handler->gemm.configure(&handler->input_tensor,
                                &handler->weight_tensor, nullptr,
                                &handler->output_tensor,
                                1.0f, 0.0f, gemm_info);

        handler->cached_M = M;
    }

    // 使用 import_memory 直接引用 PyTorch 张量的内存，避免拷贝
    handler->input_tensor.allocator()->import_memory(input.data_ptr());
    handler->output_tensor.allocator()->import_memory(output.data_ptr());

    // 执行 GEMM
    handler->gemm.run();

    // 处理偏置
    if (bias.has_value() && bias.value().defined()) {
        output.add_(bias.value());
    }

    // 释放 import_memory 的引用（不会释放底层 PyTorch 内存）
    handler->input_tensor.allocator()->free();
    handler->output_tensor.allocator()->free();
}

// 释放 ACL GEMM handler
void release_acl_gemm_handler(int64_t handler_ptr) {
    auto *handler = reinterpret_cast<ACLGEMMHandler *>(handler_ptr);
    if (handler != nullptr) {
        delete handler;
    }
}

#else  // 非 AArch64 平台

int64_t create_acl_gemm_handler(at::Tensor weight, int64_t num_threads,
                                bool fast_math) {
    throw std::runtime_error(
        "ACL GEMM: 仅在 AArch64 平台上可用");
}

void acl_gemm(at::Tensor output, at::Tensor input,
              c10::optional<at::Tensor> bias, int64_t handler_ptr) {
    throw std::runtime_error(
        "ACL GEMM: 仅在 AArch64 平台上可用");
}

void release_acl_gemm_handler(int64_t handler_ptr) {
    throw std::runtime_error(
        "ACL GEMM: 仅在 AArch64 平台上可用");
}

#endif  // __aarch64__
