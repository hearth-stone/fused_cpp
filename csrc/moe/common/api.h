// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__aarch64__)
std::tuple<std::string, std::vector<std::pair<int64_t, int64_t>>, std::vector<std::pair<int64_t, int64_t>>>
fused_moe_test_split_plan(std::string stage, int64_t M, int64_t K, int64_t N, int64_t group_size);
at::Tensor fused_moe_test_single_thread_gemm(at::Tensor A, at::Tensor B, c10::optional<at::Tensor> bias);
at::Tensor fused_moe_test_pack_interleaved_gemm(at::Tensor A, at::Tensor w13);
at::Tensor fused_moe_test_fused_w13_linear(at::Tensor A, at::Tensor w13);
at::Tensor fused_moe_test_fused_w13_silu(at::Tensor A, at::Tensor w13, int64_t degree);
at::Tensor fused_moe_test_team_fused_w13_silu(at::Tensor A, at::Tensor w13, int64_t group_size, int64_t degree);
at::Tensor fused_moe_test_pack_a_reorder_m8(at::Tensor A);
at::Tensor fused_moe_test_gather_pack_a_reorder_m8(at::Tensor input, at::Tensor routes, int64_t top_k, int64_t K_pad);
at::Tensor fused_moe_test_fused_w13_silu_packc(at::Tensor A, at::Tensor w13, int64_t degree);
at::Tensor fused_moe_test_fused_w13_silu_packc_tail(at::Tensor A, at::Tensor w13, int64_t degree);
at::Tensor fused_moe_test_team_gemm(at::Tensor A, at::Tensor B, int64_t group_size, std::string split,
                                    c10::optional<at::Tensor> bias);
std::vector<double> fused_moe_bench_team_gemm(at::Tensor A, at::Tensor B, int64_t group_size, std::string split,
                                              c10::optional<at::Tensor> bias, int64_t warmup, int64_t runs);
std::vector<double> fused_moe_bench_sve_jit_w13_gemm(at::Tensor A, at::Tensor w13_packed, int64_t K, int64_t N,
                                                     int64_t n_tile, int64_t n_ranges, int64_t warmup, int64_t runs,
                                                     int64_t probe_mode);
std::vector<double> fused_moe_bench_fused_w13_silu_packc_tail(at::Tensor A, at::Tensor w13, int64_t degree,
                                                              int64_t mode, int64_t warmup, int64_t runs);
#endif

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor w13_weight, at::Tensor w2_weight, bool fuse_silu,
                                     std::string backend_name);
at::Tensor fused_moe_bf16_tiled(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                at::Tensor topk_ids, c10::optional<at::Tensor> w13_bias,
                                c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                int64_t weight_window_bytes, c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_scheduled(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                          at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                          at::Tensor topk_ids, at::Tensor wave_offsets, at::Tensor team_expert_ids,
                                          at::Tensor team_threads, c10::optional<at::Tensor> thread_cpu_ids,
                                          c10::optional<at::Tensor> w13_bias, c10::optional<at::Tensor> w2_bias,
                                          int64_t num_threads, std::string activation, int64_t global_num_experts,
                                          bool skip_weighted, bool fuse_silu, int64_t silu_poly_degree,
                                          int64_t gemm_backend, int64_t backend_n_tile, int64_t weight_window_bytes,
                                          c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_async(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                      at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                      at::Tensor topk_ids, at::Tensor task_expert_ids, at::Tensor task_core_begins,
                                      at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps,
                                      c10::optional<at::Tensor> thread_cpu_ids, c10::optional<at::Tensor> w13_bias,
                                      c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                      int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                      int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                      int64_t w13_split, int64_t weight_window_bytes, c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_vllm_staged(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, c10::optional<at::Tensor> thread_cpu_ids,
    int64_t num_threads, int64_t global_num_experts, bool fuse_silu, int64_t silu_poly_degree, int64_t gemm_backend,
    int64_t backend_n_tile, c10::optional<at::Tensor> out);
