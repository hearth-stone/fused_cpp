// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

bool deepseek_v4_inv_rope_woa_available();
std::tuple<at::Tensor, int64_t, int64_t, int64_t> deepseek_v4_inv_rope_woa_prepare(
    at::Tensor wo_a_weight, int64_t n_groups, int64_t heads_per_group, int64_t head_dim, int64_t rope_dim,
    std::string backend);
at::Tensor deepseek_v4_inv_rope_grouped_woa(
    at::Tensor o, at::Tensor positions, at::Tensor cos_sin_cache, at::Tensor packed_weight, int64_t n_groups,
    int64_t heads_per_group, int64_t head_dim, int64_t rope_dim, int64_t output_rank,
    c10::optional<at::Tensor> core_ids, c10::optional<at::Tensor> out);

#if defined(__aarch64__)
std::tuple<std::string, std::vector<std::pair<int64_t, int64_t>>, std::vector<std::pair<int64_t, int64_t>>>
fused_moe_test_split_plan(std::string stage, int64_t M, int64_t K, int64_t N, int64_t group_size);
std::tuple<int64_t, int64_t, int64_t, std::vector<std::vector<std::pair<int64_t, int64_t>>>>
fused_moe_test_stage_window_plan(int64_t N, int64_t n_tile, int64_t threads, int64_t window_tiles);
at::Tensor fused_moe_test_single_thread_gemm(at::Tensor A, at::Tensor B, c10::optional<at::Tensor> bias);
at::Tensor fused_moe_test_pack_interleaved_gemm(at::Tensor A, at::Tensor w13);
at::Tensor fused_moe_test_fused_w13_linear(at::Tensor A, at::Tensor w13);
at::Tensor fused_moe_test_fused_w13_silu(at::Tensor A, at::Tensor w13, int64_t degree);
at::Tensor fused_moe_test_team_fused_w13_silu(at::Tensor A, at::Tensor w13, int64_t group_size, int64_t degree);
at::Tensor fused_moe_test_pack_a_reorder_m8(at::Tensor A);
at::Tensor fused_moe_test_gather_pack_a_reorder_m8(at::Tensor input, at::Tensor routes, int64_t top_k, int64_t K_pad);
std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>> fused_moe_test_gather_pack_a_work(
    int64_t total_rows, int64_t K_pad, int64_t group_size);
at::Tensor fused_moe_test_gather_pack_a_reorder_sve_hybrid(at::Tensor input, at::Tensor routes, int64_t top_k,
                                                           int64_t K_pad, int64_t group_size);
at::Tensor fused_moe_test_fused_w13_silu_packc(at::Tensor A, at::Tensor w13, int64_t degree);
at::Tensor fused_moe_test_fused_w13_silu_packc_tail(at::Tensor A, at::Tensor w13, int64_t degree);
at::Tensor fused_moe_test_team_w13_silu_packc_window(at::Tensor A, at::Tensor w13, int64_t group_size, int64_t degree,
                                                     int64_t n_tile, int64_t window_tiles, bool use_sve);
at::Tensor fused_moe_test_team_gemm(at::Tensor A, at::Tensor B, int64_t group_size, std::string split,
                                    c10::optional<at::Tensor> bias);
std::vector<double> fused_moe_bench_team_gemm(at::Tensor A, at::Tensor B, int64_t group_size, std::string split,
                                              c10::optional<at::Tensor> bias, int64_t warmup, int64_t runs);
at::Tensor fused_moe_test_sve_packed_gemm(at::Tensor A, at::Tensor packed_B, int64_t K, int64_t N, int64_t n_tile,
                                          bool use_jit);
at::Tensor fused_moe_test_team_w2_window(at::Tensor A, at::Tensor w2_packed, int64_t K, int64_t N, int64_t group_size,
                                         int64_t n_tile, int64_t window_tiles, int64_t mode);
std::vector<std::pair<int64_t, int64_t>> fused_moe_test_w2_scatter_ranges(int64_t N, int64_t n_tile, int64_t group_size,
                                                                          int64_t local_tid, int64_t window_tiles,
                                                                          bool use_w2_n_owner);
std::vector<double> fused_moe_bench_sve_jit_w13_gemm(at::Tensor A, at::Tensor w13_packed, int64_t K, int64_t N,
                                                     int64_t n_tile, int64_t n_ranges, int64_t warmup, int64_t runs,
                                                     int64_t probe_mode, bool clamp_swiglu);
std::vector<double> fused_moe_bench_fused_w13_silu_packc_tail(at::Tensor A, at::Tensor w13, int64_t degree,
                                                              int64_t mode, int64_t warmup, int64_t runs);
#endif

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor w13_weight, at::Tensor w2_weight, bool fuse_silu,
                                     std::string backend_name);
std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>
fused_moe_w8a16_tiled_prepare_weights(at::Tensor w13_weight, at::Tensor w2_weight);
std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_routed_shared_weights(at::Tensor routed_w13_weight, at::Tensor routed_w2_weight,
                                                   at::Tensor shared_w13_weight, at::Tensor shared_w2_weight,
                                                   std::string backend_name);
at::Tensor fused_moe_bf16_tiled(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                at::Tensor topk_ids, c10::optional<at::Tensor> w13_bias,
                                c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_scheduled(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                          at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                          at::Tensor topk_ids, at::Tensor wave_offsets, at::Tensor team_expert_ids,
                                          at::Tensor team_threads, c10::optional<at::Tensor> thread_cpu_ids,
                                          c10::optional<at::Tensor> w13_bias, c10::optional<at::Tensor> w2_bias,
                                          int64_t num_threads, std::string activation, int64_t global_num_experts,
                                          bool skip_weighted, bool fuse_silu, int64_t silu_poly_degree,
                                          int64_t gemm_backend, int64_t backend_n_tile,
                                          c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_async(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                      at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                      at::Tensor topk_ids, at::Tensor task_expert_ids, at::Tensor task_core_begins,
                                      at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps,
                                      c10::optional<at::Tensor> thread_cpu_ids, c10::optional<at::Tensor> w13_bias,
                                      c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                      int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                      int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                      c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_async_plan_v2(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, at::Tensor task_expert_ids, at::Tensor task_core_begins,
    at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps, int64_t plan_version,
    int64_t execution_mode, at::Tensor task_preferred_threads, at::Tensor task_min_threads, at::Tensor task_max_threads,
    at::Tensor task_allowed_thread_offsets, at::Tensor task_allowed_threads, at::Tensor task_placement_modes,
    at::Tensor task_numa_nodes, at::Tensor task_stage_ids, at::Tensor task_resize_points,
    at::Tensor task_range_granularities, c10::optional<at::Tensor> task_w13_window_tiles,
    c10::optional<at::Tensor> task_w2_window_tiles, c10::optional<at::Tensor> thread_cpu_ids,
    c10::optional<at::Tensor> w13_bias, c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
    int64_t global_num_experts, bool skip_weighted, bool fuse_silu, int64_t silu_poly_degree, int64_t gemm_backend,
    int64_t backend_n_tile, c10::optional<at::Tensor> out, int64_t early_merge);
at::Tensor fused_moe_w8a16_tiled_async_plan_v2(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w13_scales,
    at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor w2_scales, at::Tensor topk_weights,
    at::Tensor topk_ids, at::Tensor task_expert_ids, at::Tensor task_core_begins, at::Tensor task_threads,
    at::Tensor task_dep_offsets, at::Tensor task_deps, int64_t plan_version, int64_t execution_mode,
    at::Tensor task_preferred_threads, at::Tensor task_min_threads, at::Tensor task_max_threads,
    at::Tensor task_allowed_thread_offsets, at::Tensor task_allowed_threads, at::Tensor task_placement_modes,
    at::Tensor task_numa_nodes, at::Tensor task_stage_ids, at::Tensor task_resize_points,
    at::Tensor task_range_granularities, c10::optional<at::Tensor> task_w13_window_tiles,
    c10::optional<at::Tensor> task_w2_window_tiles, c10::optional<at::Tensor> thread_cpu_ids, int64_t num_threads,
    std::string activation, int64_t global_num_experts, bool skip_weighted, int64_t silu_poly_degree,
    int64_t backend_n_tile, c10::optional<at::Tensor> out, int64_t early_merge, bool cache_dequant);
at::Tensor fused_moe_bf16_tiled_planned_staged(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, at::Tensor w13_task_expert_ids,
    at::Tensor w13_task_core_begins, at::Tensor w13_task_threads, at::Tensor w13_task_dep_offsets,
    at::Tensor w13_task_deps, int64_t w13_execution_mode, at::Tensor w13_task_placement_modes,
    at::Tensor w2_task_expert_ids, at::Tensor w2_task_core_begins,
    at::Tensor w2_task_threads, at::Tensor w2_task_dep_offsets, at::Tensor w2_task_deps, int64_t w2_execution_mode,
    at::Tensor w2_task_placement_modes,
    c10::optional<at::Tensor> thread_cpu_ids, int64_t num_threads, int64_t global_num_experts, bool fuse_silu,
    int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile, c10::optional<at::Tensor> out);
at::Tensor fused_moe_bf16_tiled_vllm_staged(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, c10::optional<at::Tensor> thread_cpu_ids,
    int64_t num_threads, int64_t global_num_experts, bool fuse_silu, int64_t silu_poly_degree, int64_t gemm_backend,
    int64_t backend_n_tile, c10::optional<at::Tensor> out);
at::Tensor shared_mlp_bf16_tiled(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, c10::optional<at::Tensor> thread_cpu_ids, int64_t num_threads, bool fuse_silu,
    int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile, c10::optional<at::Tensor> out,
    bool clamp_swiglu);
