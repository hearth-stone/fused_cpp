#include <arm_sve.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 常数定义
#define LOG2_E    1.4426950408889634f  // log2(e)
#define LN2       0.6931471805599453f  // ln(2)
#define INV_LN2   1.4426950408889634f  // 1/ln(2)

// =============================================================================
// 方法1: ARM SVE Intrinsic 实现 (使用FEXPA指令)
// =============================================================================
svfloat32_t sve_exp_fexpa(svfloat32_t x, svbool_t pg) {
    // 范围约简: x = n*ln(2) + r, 其中 |r| < ln(2)/2
    svfloat32_t n_float = svrinta_f32_x(pg, svmul_f32_x(pg, x, svdup_f32(INV_LN2)));
    svfloat32_t r = svmls_f32_x(pg, x, n_float, svdup_f32(LN2));
    
    // 将n转换为整数用于后续的指数调整
    svint32_t n_int = svcvt_s32_f32_x(pg, n_float);
    
    // 使用FEXPA指令计算初始近似值
    // FEXPA需要特定格式的输入，这里我们准备r的缩放版本
    svfloat32_t scaled_r = svmul_f32_x(pg, r, svdup_f32(64.0f));
    svfloat32_t fexpa_result = svexpa_f32(scaled_r);
    
    // 多项式近似修正 (简化的泰勒级数)
    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
    svfloat32_t r2 = svmul_f32_x(pg, r, r);
    svfloat32_t r3 = svmul_f32_x(pg, r2, r);
    svfloat32_t r4 = svmul_f32_x(pg, r2, r2);
    
    svfloat32_t poly = svdup_f32(1.0f);
    poly = svmla_f32_x(pg, poly, r, svdup_f32(1.0f));
    poly = svmla_f32_x(pg, poly, r2, svdup_f32(0.5f));
    poly = svmla_f32_x(pg, poly, r3, svdup_f32(1.0f/6.0f));
    poly = svmla_f32_x(pg, poly, r4, svdup_f32(1.0f/24.0f));
    
    // 组合FEXPA结果和多项式近似
    svfloat32_t exp_r = svmul_f32_x(pg, fexpa_result, poly);
    
    // 应用指数调整: result = exp_r * 2^n
    svfloat32_t result = svscale_f32_x(pg, exp_r, n_int);
    
    return result;
}

// =============================================================================
// 方法2: ARM汇编实现 (内联汇编)
// =============================================================================
void asm_exp_vector(const float* input, float* output, int count) {
    __asm__ volatile (
        "ptrue  p0.s                    \n"  // 设置谓词寄存器
        "mov    x3, %2                  \n"  // 计数器
        "mov    x1, %0                  \n"  // 输入指针
        "mov    x2, %1                  \n"  // 输出指针
        
        // 常数准备
        "fmov   z30.s, #1.442695        \n"  // 1/ln(2)
        "fmov   z29.s, #0.693147        \n"  // ln(2)
        "fmov   z28.s, #1.0             \n"  // 常数1
        "fmov   z27.s, #0.5             \n"  // 1/2
        "mov    z26.s, #0x3e2aaaab      \n"  // 1/6
        "mov    z25.s, #0x3d2aaaab      \n"  // 1/24
        
        "loop%=:                        \n"
        // 加载输入向量
        "ld1w   {z0.s}, p0/z, [x1]     \n"
        
        // 范围约简
        "fmul   z1.s, z0.s, z30.s      \n"  // x * (1/ln2)
        "frinta z1.s, p0/m, z1.s       \n"  // round(x * (1/ln2))
        "fmls   z0.s, p0/m, z1.s, z29.s \n"  // r = x - n*ln2
        
        // 转换n为整数
        "fcvtzs z2.s, p0/m, z1.s       \n"
        
        // 计算多项式 exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
        "fmul   z3.s, z0.s, z0.s       \n"  // r²
        "fmul   z4.s, z3.s, z0.s       \n"  // r³
        "fmul   z5.s, z3.s, z3.s       \n"  // r⁴
        
        "fmov   z6.s, z28.s            \n"  // result = 1
        "fmla   z6.s, p0/m, z0.s, z28.s \n"  // result += r
        "fmla   z6.s, p0/m, z3.s, z27.s \n"  // result += r²/2
        "fmla   z6.s, p0/m, z4.s, z26.s \n"  // result += r³/6
        "fmla   z6.s, p0/m, z5.s, z25.s \n"  // result += r⁴/24
        
        // 应用2^n缩放
        "fscale z6.s, p0/m, z6.s, z2.s \n"
        
        // 存储结果
        "st1w   {z6.s}, p0, [x2]       \n"
        
        // 更新指针和计数器
        "incw   x1                      \n"
        "incw   x2                      \n"
        "decw   x3                      \n"
        "cbnz   x3, loop%=              \n"
        
        :
        : "r"(input), "r"(output), "r"(count)
        : "x1", "x2", "x3", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6",
          "z25", "z26", "z27", "z28", "z29", "z30", "memory"
    );
}

// =============================================================================
// 方法3: 普通C实现 (标量版本)
// =============================================================================
void scalar_exp_vector(const float* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        float x = input[i];
        
        // 处理特殊情况
        if (x > 88.0f) {
            output[i] = INFINITY;
            continue;
        }
        if (x < -87.0f) {
            output[i] = 0.0f;
            continue;
        }
        
        // 范围约简: x = n*ln(2) + r
        float n_float = roundf(x * INV_LN2);
        float r = x - n_float * LN2;
        int n = (int)n_float;
        
        // 多项式近似 exp(r)
        float r2 = r * r;
        float r3 = r2 * r;
        float r4 = r2 * r2;
        float r5 = r4 * r;
        float r6 = r4 * r2;
        
        float poly = 1.0f + r + 0.5f * r2 + (1.0f/6.0f) * r3 + 
                     (1.0f/24.0f) * r4 + (1.0f/120.0f) * r5 + 
                     (1.0f/720.0f) * r6;
        
        // 应用2^n缩放
        // 使用位操作实现2^n乘法
        union { float f; int i; } scale;
        scale.i = (127 + n) << 23;  // 构造2^n的IEEE 754表示
        
        output[i] = poly * scale.f;
    }
}

// =============================================================================
// 精度分析结构体
// =============================================================================
typedef struct {
    float max_abs_error;
    float max_rel_error;
    float avg_abs_error;
    float avg_rel_error;
    double sum_abs_error;
    double sum_rel_error;
    int total_samples;
    int inf_count;
    int nan_count;
} ErrorStats;

void init_error_stats(ErrorStats* stats) {
    memset(stats, 0, sizeof(ErrorStats));
}

void update_error_stats(ErrorStats* stats, float computed, float reference) {
    if (isnanf(computed) || isnanf(reference)) {
        stats->nan_count++;
        return;
    }
    
    if (isinff(computed) || isinff(reference)) {
        stats->inf_count++;
        return;
    }
    
    float abs_error = fabsf(computed - reference);
    float rel_error = (reference != 0.0f) ? abs_error / fabsf(reference) : 0.0f;
    
    stats->max_abs_error = fmaxf(stats->max_abs_error, abs_error);
    stats->max_rel_error = fmaxf(stats->max_rel_error, rel_error);
    stats->sum_abs_error += abs_error;
    stats->sum_rel_error += rel_error;
    stats->total_samples++;
}

void finalize_error_stats(ErrorStats* stats) {
    if (stats->total_samples > 0) {
        stats->avg_abs_error = stats->sum_abs_error / stats->total_samples;
        stats->avg_rel_error = stats->sum_rel_error / stats->total_samples;
    }
}

void print_error_stats(const char* name, const ErrorStats* stats) {
    printf("%-12s: ", name);
    printf("Max Abs: %8.2e, Max Rel: %8.2e, ", stats->max_abs_error, stats->max_rel_error);
    printf("Avg Abs: %8.2e, Avg Rel: %8.2e", stats->avg_abs_error, stats->avg_rel_error);
    if (stats->inf_count > 0 || stats->nan_count > 0) {
        printf(" [INF: %d, NaN: %d]", stats->inf_count, stats->nan_count);
    }
    printf("\n");
}

// =============================================================================
// 综合测试函数
// =============================================================================
void comprehensive_accuracy_test() {
    printf("\n=== Comprehensive Accuracy Test ===\n");
    
    // 测试不同的输入范围
    struct {
        const char* name;
        float start;
        float end;
        int count;
    } test_ranges[] = {
        {"Small values", -1.0f, 1.0f, 1000},
        {"Medium values", -10.0f, 10.0f, 2000},
        {"Large positive", 10.0f, 88.0f, 1000},
        {"Large negative", -88.0f, -10.0f, 1000},
        {"Edge cases", -100.0f, 100.0f, 500},
        {"Random", -50.0f, 50.0f, 5000}
    };
    
    int num_ranges = sizeof(test_ranges) / sizeof(test_ranges[0]);
    
    for (int range_idx = 0; range_idx < num_ranges; range_idx++) {
        printf("\nTesting range: %s [%.1f, %.1f] with %d samples\n", 
               test_ranges[range_idx].name,
               test_ranges[range_idx].start, 
               test_ranges[range_idx].end,
               test_ranges[range_idx].count);
        printf("%-12s  %8s %8s %8s %8s %s\n", 
               "Method", "Max Abs", "Max Rel", "Avg Abs", "Avg Rel", "Special");
        printf("%-12s  %8s %8s %8s %8s %s\n", 
               "------", "-------", "-------", "-------", "-------", "-------");
        
        int count = test_ranges[range_idx].count;
        float* input = aligned_alloc(64, count * sizeof(float));
        float* output_sve = aligned_alloc(64, count * sizeof(float));
        float* output_asm = aligned_alloc(64, count * sizeof(float));
        float* output_scalar = aligned_alloc(64, count * sizeof(float));
        float* output_ref = aligned_alloc(64, count * sizeof(float));
        
        // 生成测试数据
        float range_span = test_ranges[range_idx].end - test_ranges[range_idx].start;
        for (int i = 0; i < count; i++) {
            if (strcmp(test_ranges[range_idx].name, "Random") == 0) {
                // 随机分布
                input[i] = test_ranges[range_idx].start + 
                          ((float)rand() / RAND_MAX) * range_span;
            } else {
                // 均匀分布
                input[i] = test_ranges[range_idx].start + 
                          ((float)i / (count - 1)) * range_span;
            }
            output_ref[i] = expf(input[i]);  // 标准库参考
        }
        
        // 执行各种实现
        if (svcntw() > 0) {
            svbool_t pg = svptrue_b32();
            for (int i = 0; i < count; i += svcntw()) {
                svfloat32_t vec_in = svld1_f32(pg, &input[i]);
                svfloat32_t vec_out = sve_exp_fexpa(vec_in, pg);
                svst1_f32(pg, &output_sve[i], vec_out);
            }
        }
        
        asm_exp_vector(input, output_asm, count);
        scalar_exp_vector(input, output_scalar, count);
        
        // 计算误差统计
        ErrorStats stats_sve, stats_asm, stats_scalar;
        init_error_stats(&stats_sve);
        init_error_stats(&stats_asm);
        init_error_stats(&stats_scalar);
        
        for (int i = 0; i < count; i++) {
            if (svcntw() > 0) {
                update_error_stats(&stats_sve, output_sve[i], output_ref[i]);
            }
            update_error_stats(&stats_asm, output_asm[i], output_ref[i]);
            update_error_stats(&stats_scalar, output_scalar[i], output_ref[i]);
        }
        
        finalize_error_stats(&stats_sve);
        finalize_error_stats(&stats_asm);
        finalize_error_stats(&stats_scalar);
        
        // 打印结果
        if (svcntw() > 0) {
            print_error_stats("SVE FEXPA", &stats_sve);
        }
        print_error_stats("ASM", &stats_asm);
        print_error_stats("Scalar", &stats_scalar);
        
        // 清理内存
        free(input);
        free(output_sve);
        free(output_asm);
        free(output_scalar);
        free(output_ref);
    }
}

// =============================================================================
// 特殊值测试
// =============================================================================
void special_values_test() {
    printf("\n=== Special Values Test ===\n");
    
    float special_inputs[] = {
        0.0f, -0.0f,           // 零值
        1.0f, -1.0f,           // 单位值
        88.0f, 89.0f,          // 接近溢出
        -87.0f, -88.0f,        // 接近下溢
        INFINITY, -INFINITY,    // 无穷大
        NAN,                   // NaN
        0.5f, -0.5f,           // 常用值
        2.0f, -2.0f,
        10.0f, -10.0f,
        50.0f, -50.0f
    };
    
    int num_special = sizeof(special_inputs) / sizeof(special_inputs[0]);
    
    printf("%-10s %-12s %-12s %-12s %-12s\n", 
           "Input", "expf(x)", "SVE", "ASM", "Scalar");
    printf("%-10s %-12s %-12s %-12s %-12s\n", 
           "-----", "-------", "---", "---", "------");
    
    for (int i = 0; i < num_special; i++) {
        float input = special_inputs[i];
        float ref = expf(input);
        
        // 单个值测试
        float sve_result = 0.0f, asm_result = 0.0f, scalar_result = 0.0f;
        
        if (svcntw() > 0) {
            svbool_t pg = svptrue_b32();
            svfloat32_t vec_in = svdup_f32(input);
            svfloat32_t vec_out = sve_exp_fexpa(vec_in, pg);
            sve_result = svlastb_f32(pg, vec_out);
        }
        
        asm_exp_vector(&input, &asm_result, 1);
        scalar_exp_vector(&input, &scalar_result, 1);
        
        // 格式化输出
        char input_str[12], ref_str[12], sve_str[12], asm_str[12], scalar_str[12];
        
        if (isnanf(input)) strcpy(input_str, "NaN");
        else if (isinff(input)) strcpy(input_str, input > 0 ? "+Inf" : "-Inf");
        else snprintf(input_str, sizeof(input_str), "%.1f", input);
        
        if (isnanf(ref)) strcpy(ref_str, "NaN");
        else if (isinff(ref)) strcpy(ref_str, ref > 0 ? "+Inf" : "-Inf");
        else if (ref == 0.0f) strcpy(ref_str, "0.0");
        else snprintf(ref_str, sizeof(ref_str), "%.4e", ref);
        
        // 类似地格式化其他结果...
        if (svcntw() > 0) {
            if (isnanf(sve_result)) strcpy(sve_str, "NaN");
            else if (isinff(sve_result)) strcpy(sve_str, sve_result > 0 ? "+Inf" : "-Inf");
            else if (sve_result == 0.0f) strcpy(sve_str, "0.0");
            else snprintf(sve_str, sizeof(sve_str), "%.4e", sve_result);
        } else {
            strcpy(sve_str, "N/A");
        }
        
        if (isnanf(asm_result)) strcpy(asm_str, "NaN");
        else if (isinff(asm_result)) strcpy(asm_str, asm_result > 0 ? "+Inf" : "-Inf");
        else if (asm_result == 0.0f) strcpy(asm_str, "0.0");
        else snprintf(asm_str, sizeof(asm_str), "%.4e", asm_result);
        
        if (isnanf(scalar_result)) strcpy(scalar_str, "NaN");
        else if (isinff(scalar_result)) strcpy(scalar_str, scalar_result > 0 ? "+Inf" : "-Inf");
        else if (scalar_result == 0.0f) strcpy(scalar_str, "0.0");
        else snprintf(scalar_str, sizeof(scalar_str), "%.4e", scalar_result);
        
        printf("%-10s %-12s %-12s %-12s %-12s\n", 
               input_str, ref_str, sve_str, asm_str, scalar_str);
    }
}

// =============================================================================
// ULP (Units in the Last Place) 误差分析
// =============================================================================
int compute_ulp_error(float computed, float reference) {
    if (computed == reference) return 0;
    if (isnanf(computed) || isnanf(reference)) return INT_MAX;
    if (isinff(computed) || isinff(reference)) return INT_MAX;
    
    union { float f; uint32_t i; } comp, ref;
    comp.f = computed;
    ref.f = reference;
    
    // 处理符号不同的情况
    if ((comp.i ^ ref.i) & 0x80000000) return INT_MAX;
    
    return abs((int)comp.i - (int)ref.i);
}

void ulp_analysis() {
    printf("\n=== ULP Error Analysis ===\n");
    
    const int count = 10000;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output_sve = aligned_alloc(64, count * sizeof(float));
    float* output_asm = aligned_alloc(64, count * sizeof(float));
    float* output_scalar = aligned_alloc(64, count * sizeof(float));
    float* output_ref = aligned_alloc(64, count * sizeof(float));
    
    // 生成测试数据 (-20 到 20 的均匀分布)
    for (int i = 0; i < count; i++) {
        input[i] = -20.0f + (40.0f * i) / (count - 1);
        output_ref[i] = expf(input[i]);
    }
    
    // 执行各种实现
    if (svcntw() > 0) {
        svbool_t pg = svptrue_b32();
        for (int i = 0; i < count; i += svcntw()) {
            svfloat32_t vec_in = svld1_f32(pg, &input[i]);
            svfloat32_t vec_out = sve_exp_fexpa(vec_in, pg);
            svst1_f32(pg, &output_sve[i], vec_out);
        }
    }
    
    asm_exp_vector(input, output_asm, count);
    scalar_exp_vector(input, output_scalar, count);
    
    // ULP误差分析
    int ulp_buckets[6] = {0}; // [0], [1], [2-3], [4-7], [8-15], [16+]
    int max_ulp_sve = 0, max_ulp_asm = 0, max_ulp_scalar = 0;
    
    printf("%-12s %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "Method", "0 ULP", "1 ULP", "2-3 ULP", "4-7 ULP", "8-15 ULP", "16+ ULP");
    printf("%-12s %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "------", "-----", "-----", "-------", "-------", "--------", "-------");
    
    // 分析每种方法的ULP分布
    const char* method_names[] = {"SVE FEXPA", "ASM", "Scalar"};
    float* outputs[] = {output_sve, output_asm, output_scalar};
    
    for (int method = 0; method < 3; method++) {
        if (method == 0 && svcntw() == 0) continue; // 跳过SVE如果不支持
        
        memset(ulp_buckets, 0, sizeof(ulp_buckets));
        int max_ulp = 0;
        
        for (int i = 0; i < count; i++) {
            int ulp_error = compute_ulp_error(outputs[method][i], output_ref[i]);
            if (ulp_error == INT_MAX) continue; // 跳过特殊值
            
            max_ulp = (ulp_error > max_ulp) ? ulp_error : max_ulp;
            
            if (ulp_error == 0) ulp_buckets[0]++;
            else if (ulp_error == 1) ulp_buckets[1]++;
            else if (ulp_error <= 3) ulp_buckets[2]++;
            else if (ulp_error <= 7) ulp_buckets[3]++;
            else if (ulp_error <= 15) ulp_buckets[4]++;
            else ulp_buckets[5]++;
        }
        
        printf("%-12s %-8d %-8d %-8d %-8d %-8d %-8d (Max: %d)\n",
               method_names[method],
               ulp_buckets[0], ulp_buckets[1], ulp_buckets[2],
               ulp_buckets[3], ulp_buckets[4], ulp_buckets[5], max_ulp);
    }
    
    // 清理内存
    free(input);
    free(output_sve);
    free(output_asm);
    free(output_scalar);
    free(output_ref);
}

// =============================================================================
// 主测试函数
// =============================================================================
void test_exp_implementations() {
    printf("Testing single-precision vectorized exp() implementations\n");
    printf("=========================================================\n");
    
    // 基本精度测试
    comprehensive_accuracy_test();
    
    // 特殊值测试
    special_values_test();
    
    // ULP误差分析
    ulp_analysis();
    
    printf("\n=== Summary ===\n");
    printf("All implementations have been tested against the standard expf() function.\n");
    printf("Check the results above for accuracy comparison in different ranges.\n");
}

int main() {
    printf("Single-precision Vectorized exp() Implementation Comparison\n");
    printf("===========================================================\n");
    printf("Comparing custom implementations against standard expf()\n\n");
    
    // 设置随机种子
    srand(12345);
    
    test_exp_implementations();
    
    return 0;
}

// =============================================================================
// 编译说明:
// gcc -march=armv8-a+sve -O3 -ffast-math exp_vector.c -o exp_vector -lm
//
// 注意事项:
// 1. SVE intrinsic需要支持SVE的ARM处理器和编译器
// 2. FEXPA指令在某些实现中可能不可用，需要检查FEAT_SVE_EXP支持
// 3. 内联汇编语法可能因编译器而异
// 4. 生产代码中应该添加更多的边界检查和错误处理
// =============================================================================