#include <torch/extension.h>
#include <pybind11/stl.h>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "deepseek_v4_q_norm_rope_sve.h"

py::object flash_mla_sparse_fwd(at::Tensor q, at::Tensor kv, at::Tensor indices, double sm_scale,
                                c10::optional<int64_t> d_v, c10::optional<at::Tensor> attn_sink,
                                c10::optional<at::Tensor> topk_length, c10::optional<at::Tensor> out,
                                bool return_stats);

// OMP runtime info forward declarations — omp_info.cpp
std::map<std::string, std::string> get_omp_runtime_info();
bool has_openmp();

// I8 GEMM declarations - i8gemm.cpp
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, std::string> i8gemm_prepare(
    at::Tensor weight, at::Tensor weight_scale, std::string backend);
void i8gemm_dynamic_scaled_mm(at::Tensor output, at::Tensor input, at::Tensor packed_weight, at::Tensor weight_scale,
                              c10::optional<at::Tensor> bias, int64_t K, int64_t N, int64_t Kp, int64_t Np,
                              int64_t nthreads);
void i8gemm_dynamic_scaled_mm_pair(at::Tensor first_output, at::Tensor second_output, at::Tensor input,
                                   at::Tensor first_packed_weight, at::Tensor first_weight_scale, int64_t K,
                                   int64_t first_N, int64_t Kp, int64_t first_Np, at::Tensor second_packed_weight,
                                   at::Tensor second_weight_scale, int64_t second_N, int64_t second_Np,
                                   int64_t nthreads);

// DeepSeek V4 mHC SVE narrow projection declarations.
bool deepseek_v4_mhc_sve_projection_available();
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> deepseek_v4_mhc_sve_projection(at::Tensor residual,
                                                                                    at::Tensor packed_b,
                                                                                    int64_t num_threads,
                                                                                    int64_t b_window_bytes);
std::tuple<at::Tensor, at::Tensor, at::Tensor> deepseek_v4_mhc_sve_control_postprocess(
    at::Tensor mixes, at::Tensor sqrsum, at::Tensor hc_scale, at::Tensor hc_base, int64_t rms_elements,
    double rms_eps, double pre_eps, double post_multiplier, double sinkhorn_eps, int64_t sinkhorn_repeat,
    int64_t num_threads);
std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t> deepseek_v4_mhc_sve_projection_control(
    at::Tensor residual, at::Tensor packed_b, at::Tensor hc_scale, at::Tensor hc_base, double rms_eps,
    double pre_eps, double post_multiplier, double sinkhorn_eps, int64_t sinkhorn_repeat, int64_t num_threads,
    int64_t b_window_bytes);
at::Tensor deepseek_v4_mhc_sve_pre_apply_rmsnorm(at::Tensor residual, at::Tensor pre_mix, at::Tensor norm_weight,
                                                 double norm_eps, int64_t num_threads);
at::Tensor deepseek_v4_mhc_sve_post(at::Tensor layer_output, at::Tensor residual, at::Tensor post_mix,
                                    at::Tensor comb_mix, int64_t num_threads);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_mhc_sve_post_head_rmsnorm(
    at::Tensor layer_output, at::Tensor residual, at::Tensor post_mix, at::Tensor comb_mix, at::Tensor packed_head,
    at::Tensor head_scale, at::Tensor head_base, at::Tensor norm_weight, double rms_eps, double head_eps,
    double norm_eps, int64_t num_threads);
std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t> deepseek_v4_mhc_sve_pre_rmsnorm(
    at::Tensor residual, at::Tensor packed_b, at::Tensor hc_scale, at::Tensor hc_base, at::Tensor norm_weight,
    double rms_eps, double pre_eps, double post_multiplier, double sinkhorn_eps, int64_t sinkhorn_repeat,
    double norm_eps, int64_t num_threads, int64_t b_window_bytes);

// BF16 GEMM declarations - bf16_linear.cpp
std::tuple<at::Tensor, int64_t, int64_t, int64_t> bf16_linear_prepare_weight(at::Tensor weight);
at::Tensor bf16_linear_prepacked_to_dtype(at::Tensor input, at::Tensor packed_weight, int64_t K, int64_t N, int64_t Np,
                                          bool output_bf16, int64_t nthreads);
at::Tensor bf16_linear_to_dtype(at::Tensor input, at::Tensor weight, bool output_bf16, int64_t nthreads);

// DeepSeek V4 attn_gemm_parallel_execute fused GEMM declarations
std::tuple<at::Tensor, int64_t, int64_t>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare(at::Tensor weight);
std::tuple<at::Tensor, int64_t, int64_t> deepseek_v4_post_gemm_prepare(at::Tensor weight);
at::Tensor fused_wqa_wkv_fused(at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K,
                               int64_t fused_wqa_wkv_N);
at::Tensor fused_wqa_wkv_fused_mt(at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K,
                                  int64_t fused_wqa_wkv_N, std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N);
std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N, std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_qkv_rmsnorm_fused(at::Tensor hidden_states,
                                                                   at::Tensor fused_wqa_wkv_packed,
                                                                   int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
                                                                   at::Tensor q_norm_weight, at::Tensor kv_norm_weight,
                                                                   int64_t q_lora_rank, int64_t kv_dim, double eps);
std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_qkv_rmsnorm_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim, double eps,
    std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim, double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim, double eps,
    std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N, at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank,
    int64_t kv_dim, double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N, at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank,
    int64_t kv_dim, double eps, std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage(
    at::Tensor qr, at::Tensor kv, at::Tensor kv_score, at::Tensor indexer_kv_score, at::Tensor indexer_weights,
    at::Tensor positions, at::Tensor main_wq_b_weight, at::Tensor indexer_wq_b_weight, at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache, at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape,
    at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table, at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight,
    at::Tensor indexer_ape, at::Tensor indexer_state_cache, at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices, at::Tensor indexer_block_table, at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping, at::Tensor indexer_norm_weight, at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens, at::Tensor prefill_cu_seqlen_ks, at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table, int64_t main_head_dim, double q_eps, int64_t mla_compress_ratio,
    double mla_rms_norm_eps, int64_t indexer_compress_ratio, double indexer_rms_norm_eps, int64_t topk_tokens);
at::Tensor deepseek_v4_post_gemm_dense_prepacked(at::Tensor qr, at::Tensor kv, at::Tensor positions,
                                                 at::Tensor main_wq_b_packed, int64_t main_wq_b_K, int64_t main_wq_b_N,
                                                 int64_t main_wq_b_Np, at::Tensor main_cos_sin_cache,
                                                 at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping,
                                                 int64_t main_head_dim, double q_eps);
at::Tensor deepseek_v4_post_gemm_c128a_prepacked(
    at::Tensor qr, at::Tensor kv, at::Tensor kv_score, at::Tensor positions, at::Tensor main_wq_b_packed,
    int64_t main_wq_b_K, int64_t main_wq_b_N, int64_t main_wq_b_Np, at::Tensor main_cos_sin_cache,
    at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape, at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices, at::Tensor mla_block_table,
    at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight, int64_t main_head_dim,
    double q_eps, int64_t mla_compress_ratio, double mla_rms_norm_eps);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage_prepacked(
    at::Tensor qr, at::Tensor kv, at::Tensor kv_score, at::Tensor indexer_kv_score, at::Tensor indexer_weights,
    at::Tensor positions, at::Tensor main_wq_b_packed, int64_t main_wq_b_K, int64_t main_wq_b_N, int64_t main_wq_b_Np,
    at::Tensor indexer_wq_b_packed, int64_t indexer_wq_b_K, int64_t indexer_wq_b_N, int64_t indexer_wq_b_Np,
    at::Tensor main_cos_sin_cache, at::Tensor indexer_cos_sin_cache, at::Tensor swa_kv_cache,
    at::Tensor swa_slot_mapping, at::Tensor mla_ape, at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping,
    at::Tensor mla_token_to_req_indices, at::Tensor mla_block_table, at::Tensor mla_kv_cache,
    at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight, at::Tensor indexer_ape, at::Tensor indexer_state_cache,
    at::Tensor indexer_state_slot_mapping, at::Tensor indexer_token_to_req_indices, at::Tensor indexer_block_table,
    at::Tensor indexer_kv_cache, at::Tensor indexer_kv_slot_mapping, at::Tensor indexer_norm_weight,
    at::Tensor topk_indices_buffer, at::Tensor prefill_cu_seq_lens, at::Tensor prefill_cu_seqlen_ks,
    at::Tensor prefill_cu_seqlen_ke, at::Tensor prefill_block_table, int64_t main_head_dim, double q_eps,
    int64_t mla_compress_ratio, double mla_rms_norm_eps, int64_t indexer_compress_ratio, double indexer_rms_norm_eps,
    int64_t topk_tokens);
at::Tensor deepseek_v4_post_gemm_dense_projected(at::Tensor main_q_linear, at::Tensor kv, at::Tensor positions,
                                                 at::Tensor main_cos_sin_cache, at::Tensor swa_kv_cache,
                                                 at::Tensor swa_slot_mapping, int64_t main_head_dim, double q_eps);
at::Tensor deepseek_v4_post_gemm_c128a_projected(
    at::Tensor main_q_linear, at::Tensor kv, at::Tensor kv_score, at::Tensor positions,
    at::Tensor main_cos_sin_cache, at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape,
    at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table, at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight,
    int64_t main_head_dim, double q_eps, int64_t mla_compress_ratio, double mla_rms_norm_eps);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage_projected(
    at::Tensor main_q_linear, at::Tensor indexer_q_linear, at::Tensor kv, at::Tensor kv_score,
    at::Tensor indexer_kv_score, at::Tensor indexer_weights, at::Tensor positions, at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache, at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape,
    at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table, at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight,
    at::Tensor indexer_ape, at::Tensor indexer_state_cache, at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices, at::Tensor indexer_block_table, at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping, at::Tensor indexer_norm_weight, at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens, at::Tensor prefill_cu_seqlen_ks, at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table, int64_t main_head_dim, double q_eps, int64_t mla_compress_ratio,
    double mla_rms_norm_eps, int64_t indexer_compress_ratio, double indexer_rms_norm_eps, int64_t topk_tokens);
void deepseek_v4_dequantize_and_gather_k_cache(at::Tensor out, at::Tensor k_cache, at::Tensor seq_lens,
                                               c10::optional<at::Tensor> gather_lens, at::Tensor block_table,
                                               int64_t block_size, int64_t offset);
void deepseek_v4_dequantize_and_gather_dual_k_cache(at::Tensor out, at::Tensor compressed_k_cache,
                                                    at::Tensor compressed_seq_lens, at::Tensor compressed_block_table,
                                                    int64_t compressed_block_size, int64_t compressed_offset,
                                                    bool has_compressed, at::Tensor swa_k_cache,
                                                    at::Tensor swa_seq_lens, at::Tensor swa_gather_lens,
                                                    at::Tensor swa_block_table, int64_t swa_block_size,
                                                    int64_t swa_offset);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_combine_topk_swa_indices(at::Tensor topk_indices,
                                                                        at::Tensor query_start_loc, at::Tensor seq_lens,
                                                                        at::Tensor gather_lens, int64_t window_size,
                                                                        int64_t compress_ratio, int64_t topk, int64_t M,
                                                                        int64_t N);

PYBIND11_MODULE(_C, m) {
  m.doc() = "fused_cpp C++ extension kernels";


  m.def("flash_mla_sparse_fwd", &flash_mla_sparse_fwd,
        "Sparse MLA forward. BF16 uses head-major 8x8 QK/PV when query "
        "parallelism is sufficient, with guarded 2D indexed fallback.",
        py::arg("q"), py::arg("kv"), py::arg("indices"), py::arg("sm_scale"), py::arg("d_v") = c10::nullopt,
        py::arg("attn_sink") = c10::nullopt, py::arg("topk_length") = c10::nullopt, py::arg("out") = c10::nullopt,
        py::arg("return_stats") = false, py::call_guard<py::gil_scoped_release>());

  m.def("get_omp_runtime_info", &get_omp_runtime_info,
        "Return a snapshot of the current OpenMP runtime configuration "
        "(env vars + omp_get_max_threads / num_procs).");

  m.def("has_openmp", &has_openmp, "Return True if the C++ extension was linked with OpenMP.");

  // ── I8 GEMM 接口 ──
  m.def("i8gemm_prepare", &i8gemm_prepare, "Prepare int8 [N, K] vLLM linear weight for i8 GEMM.", py::arg("weight"),
        py::arg("weight_scale"), py::arg("backend") = "auto", py::call_guard<py::gil_scoped_release>());

  m.def("i8gemm_dynamic_scaled_mm", &i8gemm_dynamic_scaled_mm,
        "Dynamic per-token scaled int8 GEMM with fp32/bf16 output.", py::arg("output"), py::arg("input"),
        py::arg("packed_weight"), py::arg("weight_scale"), py::arg("bias") = c10::nullopt, py::arg("K"), py::arg("N"),
        py::arg("Kp"), py::arg("Np"), py::arg("nthreads") = 0, py::call_guard<py::gil_scoped_release>());

  m.def("i8gemm_dynamic_scaled_mm_pair", &i8gemm_dynamic_scaled_mm_pair,
        "Run two dynamic W8A8 GEMMs while sharing one BF16-to-A8 quantization.", py::arg("first_output"),
        py::arg("second_output"), py::arg("input"), py::arg("first_packed_weight"), py::arg("first_weight_scale"),
        py::arg("K"), py::arg("first_N"), py::arg("Kp"), py::arg("first_Np"), py::arg("second_packed_weight"),
        py::arg("second_weight_scale"), py::arg("second_N"), py::arg("second_Np"), py::arg("nthreads") = 0,
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_mhc_sve_projection_available", &deepseek_v4_mhc_sve_projection_available,
        "Return whether the M8/M4 by N24 SVE FP32 projection is available.");
  m.def("deepseek_v4_mhc_sve_projection", &deepseek_v4_mhc_sve_projection,
        "Project BF16 [M,C,H] by packed FP32 [C*H,24] and return FP32 mixes/sqrsum.", py::arg("residual"),
        py::arg("packed_b"), py::arg("num_threads") = 0, py::arg("b_window_bytes") = 1 << 20,
        py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_mhc_sve_control_postprocess", &deepseek_v4_mhc_sve_control_postprocess,
        "Apply FP32 mHC pre/post sigmoid and strided-T Sinkhorn to [T,24] projection controls.", py::arg("mixes"),
        py::arg("sqrsum"), py::arg("hc_scale"), py::arg("hc_base"), py::arg("rms_elements"), py::arg("rms_eps"),
        py::arg("pre_eps"), py::arg("post_multiplier"), py::arg("sinkhorn_eps"), py::arg("sinkhorn_repeat"),
        py::arg("num_threads") = 0, py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_mhc_sve_projection_control", &deepseek_v4_mhc_sve_projection_control,
        "Run the FP32 N24 projection followed by native SVE pre/post sigmoid and strided-T Sinkhorn.",
        py::arg("residual"), py::arg("packed_b"), py::arg("hc_scale"), py::arg("hc_base"), py::arg("rms_eps"),
        py::arg("pre_eps"), py::arg("post_multiplier"), py::arg("sinkhorn_eps"), py::arg("sinkhorn_repeat"),
        py::arg("num_threads") = 0, py::arg("b_window_bytes") = 1 << 20,
        py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_mhc_sve_pre_apply_rmsnorm", &deepseek_v4_mhc_sve_pre_apply_rmsnorm,
        "Apply FP32 pre controls to four BF16 residual streams, preserve the BF16 boundary, and run RMSNorm.",
        py::arg("residual"), py::arg("pre_mix"), py::arg("norm_weight"), py::arg("norm_eps"),
        py::arg("num_threads") = 0, py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_mhc_sve_post", &deepseek_v4_mhc_sve_post,
        "Apply fixed-K4 residual mixing plus the rank-one layer-output injection and store BF16 residual.",
        py::arg("layer_output"), py::arg("residual"), py::arg("post_mix"), py::arg("comb_mix"),
        py::arg("num_threads") = 0, py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_mhc_sve_post_head_rmsnorm", &deepseek_v4_mhc_sve_post_head_rmsnorm,
        "Run native post, NEON M12xN4 HC-head projection, four-stream reduction, and final RMSNorm.",
        py::arg("layer_output"), py::arg("residual"), py::arg("post_mix"), py::arg("comb_mix"),
        py::arg("packed_head"), py::arg("head_scale"), py::arg("head_base"), py::arg("norm_weight"),
        py::arg("rms_eps"), py::arg("head_eps"), py::arg("norm_eps"), py::arg("num_threads") = 0,
        py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_mhc_sve_pre_rmsnorm", &deepseek_v4_mhc_sve_pre_rmsnorm,
        "Run SVE mHC projection, control processing, pre residual reduction, and RMSNorm.", py::arg("residual"),
        py::arg("packed_b"), py::arg("hc_scale"), py::arg("hc_base"), py::arg("norm_weight"), py::arg("rms_eps"),
        py::arg("pre_eps"), py::arg("post_multiplier"), py::arg("sinkhorn_eps"), py::arg("sinkhorn_repeat"),
        py::arg("norm_eps"), py::arg("num_threads") = 0, py::arg("b_window_bytes") = 1 << 20,
        py::call_guard<py::gil_scoped_release>());

  m.def("bf16_linear_to_dtype", &bf16_linear_to_dtype,
        "BF16 [M,K] x [N,K]^T linear using refs/i8gemm bf16gemm. "
        "The accumulator/output kernel is fp32; output_bf16 casts the "
        "visible result to bf16.",
        py::arg("input"), py::arg("weight"), py::arg("output_bf16"), py::arg("nthreads") = 0,
        py::call_guard<py::gil_scoped_release>());

  m.def("bf16_linear_prepare_weight", &bf16_linear_prepare_weight,
        "Prepack a bf16 PyTorch linear weight [N,K] for refs/i8gemm "
        "bf16gemm. Returns (packed_weight, K, N, Np).",
        py::arg("weight"), py::call_guard<py::gil_scoped_release>());

  m.def("bf16_linear_prepacked_to_dtype", &bf16_linear_prepacked_to_dtype,
        "BF16 [M,K] linear using a prepacked refs/i8gemm bf16gemm weight.", py::arg("input"), py::arg("packed_weight"),
        py::arg("K"), py::arg("N"), py::arg("Np"), py::arg("output_bf16"), py::arg("nthreads") = 0,
        py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare",
        &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare,
        "Pack one bf16 [K, N] weight for the DeepSeek V4 fused attn GEMM path.", py::arg("weight"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_post_gemm_prepare", &deepseek_v4_post_gemm_prepare,
        "Pack one bf16 [K, N] weight for the selected DeepSeek V4 post-GEMM backend.", py::arg("weight"),
        py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_fused", &fused_wqa_wkv_fused, "DeepSeek V4 dense attention input GEMM path: fused_wqa_wkv bf16.",
        py::arg("hidden_states"), py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"),
        py::arg("fused_wqa_wkv_N"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_fused_mt", &fused_wqa_wkv_fused_mt, "OpenMP DeepSeek V4 dense attention input GEMM path.",
        py::arg("hidden_states"), py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"),
        py::arg("fused_wqa_wkv_N"), py::arg("core_ids"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_fused", &fused_wqa_wkv_compressor_kv_score_fused,
        "DeepSeek V4 C128A attention input GEMM path: fused_wqa_wkv bf16 "
        "and compressor_kv_score fp32.",
        py::arg("hidden_states"), py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"),
        py::arg("fused_wqa_wkv_N"), py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"),
        py::arg("compressor_kv_score_N"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_fused_mt", &fused_wqa_wkv_compressor_kv_score_fused_mt,
        "OpenMP DeepSeek V4 C128A attention input GEMM path.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"), py::arg("compressor_kv_score_N"),
        py::arg("core_ids"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused",
        &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused,
        "Serial DeepSeek V4 C4A attn_gemm_parallel_execute fused path: "
        "fused_wqa_wkv bf16, compressor_kv_score fp32, "
        "indexer_compressor_kv_score fp32, indexer_weights_proj bf16.",
        py::arg("hidden_states"), py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"),
        py::arg("fused_wqa_wkv_N"), py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"),
        py::arg("compressor_kv_score_N"), py::arg("indexer_compressor_kv_score_packed"),
        py::arg("indexer_compressor_kv_score_K"), py::arg("indexer_compressor_kv_score_N"),
        py::arg("indexer_weights_proj_packed"), py::arg("indexer_weights_proj_K"), py::arg("indexer_weights_proj_N"),
        py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt",
        &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt,
        "OpenMP DeepSeek V4 C4A attn_gemm_parallel_execute fused path. "
        "core_ids controls thread count and per-thread CPU affinity; "
        "rows are split into contiguous ceil(M / len(core_ids)) chunks.",
        py::arg("hidden_states"), py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"),
        py::arg("fused_wqa_wkv_N"), py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"),
        py::arg("compressor_kv_score_N"), py::arg("indexer_compressor_kv_score_packed"),
        py::arg("indexer_compressor_kv_score_K"), py::arg("indexer_compressor_kv_score_N"),
        py::arg("indexer_weights_proj_packed"), py::arg("indexer_weights_proj_K"), py::arg("indexer_weights_proj_N"),
        py::arg("core_ids"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_qkv_rmsnorm_fused", &fused_wqa_wkv_qkv_rmsnorm_fused,
        "DeepSeek V4 dense attention input GEMM plus q/kv RMSNorm.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("q_norm_weight"), py::arg("kv_norm_weight"), py::arg("q_lora_rank"), py::arg("kv_dim"), py::arg("eps"),
        py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_qkv_rmsnorm_fused_mt", &fused_wqa_wkv_qkv_rmsnorm_fused_mt,
        "OpenMP DeepSeek V4 dense attention input GEMM plus q/kv RMSNorm.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("q_norm_weight"), py::arg("kv_norm_weight"), py::arg("q_lora_rank"), py::arg("kv_dim"), py::arg("eps"),
        py::arg("core_ids"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused", &fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused,
        "DeepSeek V4 C128A attention input GEMMs plus q/kv RMSNorm.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"), py::arg("compressor_kv_score_N"),
        py::arg("q_norm_weight"), py::arg("kv_norm_weight"), py::arg("q_lora_rank"), py::arg("kv_dim"), py::arg("eps"),
        py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused_mt",
        &fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused_mt,
        "OpenMP DeepSeek V4 C128A attention input GEMMs plus q/kv RMSNorm.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"), py::arg("compressor_kv_score_N"),
        py::arg("q_norm_weight"), py::arg("kv_norm_weight"), py::arg("q_lora_rank"), py::arg("kv_dim"), py::arg("eps"),
        py::arg("core_ids"), py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused",
        &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused,
        "Serial DeepSeek V4 C4A input GEMMs plus q/kv RMSNorm.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"), py::arg("compressor_kv_score_N"),
        py::arg("indexer_compressor_kv_score_packed"), py::arg("indexer_compressor_kv_score_K"),
        py::arg("indexer_compressor_kv_score_N"), py::arg("indexer_weights_proj_packed"),
        py::arg("indexer_weights_proj_K"), py::arg("indexer_weights_proj_N"), py::arg("q_norm_weight"),
        py::arg("kv_norm_weight"), py::arg("q_lora_rank"), py::arg("kv_dim"), py::arg("eps"),
        py::call_guard<py::gil_scoped_release>());

  m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused_mt",
        &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused_mt,
        "OpenMP DeepSeek V4 C4A input GEMMs plus q/kv RMSNorm.", py::arg("hidden_states"),
        py::arg("fused_wqa_wkv_packed"), py::arg("fused_wqa_wkv_K"), py::arg("fused_wqa_wkv_N"),
        py::arg("compressor_kv_score_packed"), py::arg("compressor_kv_score_K"), py::arg("compressor_kv_score_N"),
        py::arg("indexer_compressor_kv_score_packed"), py::arg("indexer_compressor_kv_score_K"),
        py::arg("indexer_compressor_kv_score_N"), py::arg("indexer_weights_proj_packed"),
        py::arg("indexer_weights_proj_K"), py::arg("indexer_weights_proj_N"), py::arg("q_norm_weight"),
        py::arg("kv_norm_weight"), py::arg("q_lora_rank"), py::arg("kv_dim"), py::arg("eps"), py::arg("core_ids"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_post_gemm_parallel_stage", &deepseek_v4_post_gemm_parallel_stage,
        "Torch-API C++ baseline for the DeepSeek V4 post-GEMM attention "
        "stage. Covers the CPU prefill semantics of the second "
        "execute_in_parallel block: main q path, MLA compressor, indexer "
        "compressor, and sparse indexer top-k.",
        py::arg("qr"), py::arg("kv"), py::arg("kv_score"), py::arg("indexer_kv_score"), py::arg("indexer_weights"),
        py::arg("positions"), py::arg("main_wq_b_weight"), py::arg("indexer_wq_b_weight"),
        py::arg("main_cos_sin_cache"), py::arg("indexer_cos_sin_cache"), py::arg("swa_kv_cache"),
        py::arg("swa_slot_mapping"), py::arg("mla_ape"), py::arg("mla_state_cache"), py::arg("mla_state_slot_mapping"),
        py::arg("mla_token_to_req_indices"), py::arg("mla_block_table"), py::arg("mla_kv_cache"),
        py::arg("mla_kv_slot_mapping"), py::arg("mla_norm_weight"), py::arg("indexer_ape"),
        py::arg("indexer_state_cache"), py::arg("indexer_state_slot_mapping"), py::arg("indexer_token_to_req_indices"),
        py::arg("indexer_block_table"), py::arg("indexer_kv_cache"), py::arg("indexer_kv_slot_mapping"),
        py::arg("indexer_norm_weight"), py::arg("topk_indices_buffer"), py::arg("prefill_cu_seq_lens"),
        py::arg("prefill_cu_seqlen_ks"), py::arg("prefill_cu_seqlen_ke"), py::arg("prefill_block_table"),
        py::arg("main_head_dim"), py::arg("q_eps"), py::arg("mla_compress_ratio"), py::arg("mla_rms_norm_eps"),
        py::arg("indexer_compress_ratio"), py::arg("indexer_rms_norm_eps"), py::arg("topk_tokens"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_post_gemm_dense_prepacked", &deepseek_v4_post_gemm_dense_prepacked,
        "DeepSeek V4 post-GEMM dense prepacked stage: main q path and SWA "
        "KV cache insertion only.",
        py::arg("qr"), py::arg("kv"), py::arg("positions"), py::arg("main_wq_b_packed"), py::arg("main_wq_b_K"),
        py::arg("main_wq_b_N"), py::arg("main_wq_b_Np"), py::arg("main_cos_sin_cache"), py::arg("swa_kv_cache"),
        py::arg("swa_slot_mapping"), py::arg("main_head_dim"), py::arg("q_eps"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_post_gemm_c128a_prepacked", &deepseek_v4_post_gemm_c128a_prepacked,
        "DeepSeek V4 post-GEMM C128A prepacked stage: dense main path plus "
        "the main MLA compressor.",
        py::arg("qr"), py::arg("kv"), py::arg("kv_score"), py::arg("positions"), py::arg("main_wq_b_packed"),
        py::arg("main_wq_b_K"), py::arg("main_wq_b_N"), py::arg("main_wq_b_Np"), py::arg("main_cos_sin_cache"),
        py::arg("swa_kv_cache"), py::arg("swa_slot_mapping"), py::arg("mla_ape"), py::arg("mla_state_cache"),
        py::arg("mla_state_slot_mapping"), py::arg("mla_token_to_req_indices"), py::arg("mla_block_table"),
        py::arg("mla_kv_cache"), py::arg("mla_kv_slot_mapping"), py::arg("mla_norm_weight"), py::arg("main_head_dim"),
        py::arg("q_eps"), py::arg("mla_compress_ratio"), py::arg("mla_rms_norm_eps"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_post_gemm_parallel_stage_prepacked", &deepseek_v4_post_gemm_parallel_stage_prepacked,
        "Torch-API C++ baseline for the DeepSeek V4 post-GEMM attention "
        "stage using prepacked bf16 weights for the two post linear layers.",
        py::arg("qr"), py::arg("kv"), py::arg("kv_score"), py::arg("indexer_kv_score"), py::arg("indexer_weights"),
        py::arg("positions"), py::arg("main_wq_b_packed"), py::arg("main_wq_b_K"), py::arg("main_wq_b_N"),
        py::arg("main_wq_b_Np"), py::arg("indexer_wq_b_packed"), py::arg("indexer_wq_b_K"), py::arg("indexer_wq_b_N"),
        py::arg("indexer_wq_b_Np"), py::arg("main_cos_sin_cache"), py::arg("indexer_cos_sin_cache"),
        py::arg("swa_kv_cache"), py::arg("swa_slot_mapping"), py::arg("mla_ape"), py::arg("mla_state_cache"),
        py::arg("mla_state_slot_mapping"), py::arg("mla_token_to_req_indices"), py::arg("mla_block_table"),
        py::arg("mla_kv_cache"), py::arg("mla_kv_slot_mapping"), py::arg("mla_norm_weight"), py::arg("indexer_ape"),
        py::arg("indexer_state_cache"), py::arg("indexer_state_slot_mapping"), py::arg("indexer_token_to_req_indices"),
        py::arg("indexer_block_table"), py::arg("indexer_kv_cache"), py::arg("indexer_kv_slot_mapping"),
        py::arg("indexer_norm_weight"), py::arg("topk_indices_buffer"), py::arg("prefill_cu_seq_lens"),
        py::arg("prefill_cu_seqlen_ks"), py::arg("prefill_cu_seqlen_ke"), py::arg("prefill_block_table"),
        py::arg("main_head_dim"), py::arg("q_eps"), py::arg("mla_compress_ratio"), py::arg("mla_rms_norm_eps"),
        py::arg("indexer_compress_ratio"), py::arg("indexer_rms_norm_eps"), py::arg("topk_tokens"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_post_gemm_dense_projected", &deepseek_v4_post_gemm_dense_projected,
        "DeepSeek V4 dense post stage from a precomputed BF16 main-Q projection.", py::arg("main_q_linear"),
        py::arg("kv"), py::arg("positions"), py::arg("main_cos_sin_cache"), py::arg("swa_kv_cache"),
        py::arg("swa_slot_mapping"), py::arg("main_head_dim"), py::arg("q_eps"),
        py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_post_gemm_c128a_projected", &deepseek_v4_post_gemm_c128a_projected,
        "DeepSeek V4 C128A post stage from a precomputed BF16 main-Q projection.", py::arg("main_q_linear"),
        py::arg("kv"), py::arg("kv_score"), py::arg("positions"), py::arg("main_cos_sin_cache"),
        py::arg("swa_kv_cache"), py::arg("swa_slot_mapping"), py::arg("mla_ape"), py::arg("mla_state_cache"),
        py::arg("mla_state_slot_mapping"), py::arg("mla_token_to_req_indices"), py::arg("mla_block_table"),
        py::arg("mla_kv_cache"), py::arg("mla_kv_slot_mapping"), py::arg("mla_norm_weight"),
        py::arg("main_head_dim"), py::arg("q_eps"), py::arg("mla_compress_ratio"), py::arg("mla_rms_norm_eps"),
        py::call_guard<py::gil_scoped_release>());
  m.def("deepseek_v4_post_gemm_parallel_stage_projected", &deepseek_v4_post_gemm_parallel_stage_projected,
        "DeepSeek V4 C4A post stage from precomputed BF16 main/indexer Q projections.",
        py::arg("main_q_linear"), py::arg("indexer_q_linear"), py::arg("kv"), py::arg("kv_score"),
        py::arg("indexer_kv_score"), py::arg("indexer_weights"), py::arg("positions"),
        py::arg("main_cos_sin_cache"), py::arg("indexer_cos_sin_cache"), py::arg("swa_kv_cache"),
        py::arg("swa_slot_mapping"), py::arg("mla_ape"), py::arg("mla_state_cache"),
        py::arg("mla_state_slot_mapping"), py::arg("mla_token_to_req_indices"), py::arg("mla_block_table"),
        py::arg("mla_kv_cache"), py::arg("mla_kv_slot_mapping"), py::arg("mla_norm_weight"),
        py::arg("indexer_ape"), py::arg("indexer_state_cache"), py::arg("indexer_state_slot_mapping"),
        py::arg("indexer_token_to_req_indices"), py::arg("indexer_block_table"), py::arg("indexer_kv_cache"),
        py::arg("indexer_kv_slot_mapping"), py::arg("indexer_norm_weight"), py::arg("topk_indices_buffer"),
        py::arg("prefill_cu_seq_lens"), py::arg("prefill_cu_seqlen_ks"), py::arg("prefill_cu_seqlen_ke"),
        py::arg("prefill_block_table"), py::arg("main_head_dim"), py::arg("q_eps"),
        py::arg("mla_compress_ratio"), py::arg("mla_rms_norm_eps"), py::arg("indexer_compress_ratio"),
        py::arg("indexer_rms_norm_eps"), py::arg("topk_tokens"), py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_dequantize_and_gather_k_cache", &deepseek_v4_dequantize_and_gather_k_cache,
        "DeepSeek V4 CPU prefill bf16 paged K-cache gather baseline. CPU "
        "stores bf16 cache directly, so this is the CPU counterpart of the "
        "GPU dequantize-and-gather op without FP8 dequantization.",
        py::arg("out"), py::arg("k_cache"), py::arg("seq_lens"), py::arg("gather_lens"), py::arg("block_table"),
        py::arg("block_size"), py::arg("offset"), py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_dequantize_and_gather_dual_k_cache", &deepseek_v4_dequantize_and_gather_dual_k_cache,
        "DeepSeek V4 CPU prefill bf16 paged K-cache dual gather. This "
        "experimental optimized entrypoint writes the compressed cache "
        "region and SWA cache region into the same workspace in one call; "
        "the single-gather op remains the compatibility baseline.",
        py::arg("out"), py::arg("compressed_k_cache"), py::arg("compressed_seq_lens"),
        py::arg("compressed_block_table"), py::arg("compressed_block_size"), py::arg("compressed_offset"),
        py::arg("has_compressed"), py::arg("swa_k_cache"), py::arg("swa_seq_lens"), py::arg("swa_gather_lens"),
        py::arg("swa_block_table"), py::arg("swa_block_size"), py::arg("swa_offset"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_combine_topk_swa_indices", &deepseek_v4_combine_topk_swa_indices,
        "DeepSeek V4 CPU prefill sparse-index combiner baseline. Combines "
        "compressed top-k indices with the local SWA window indices.",
        py::arg("topk_indices"), py::arg("query_start_loc"), py::arg("seq_lens"), py::arg("gather_lens"),
        py::arg("window_size"), py::arg("compress_ratio"), py::arg("topk"), py::arg("M"), py::arg("N"),
        py::call_guard<py::gil_scoped_release>());

  m.def("deepseek_v4_q_norm_rope_fused_sve", &::fused_cpp::deepseek_v4::q_norm_rope_fused_sve,
        "Run only the DeepSeek V4 q RMSNorm+RoPE SVE fast path in-place. "
        "Inputs must already be CPU tensors with q stride(-1)=1, positions "
        "as int64, and cos_sin_cache as contiguous float32.",
        py::arg("q"), py::arg("positions_long"), py::arg("cos_sin_f"), py::arg("eps"),
        py::call_guard<py::gil_scoped_release>());

}
