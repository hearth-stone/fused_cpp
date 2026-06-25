#include <torch/extension.h>
#include <pybind11/stl.h>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "deepseek_v4_q_norm_rope_sve.h"

// Forward declarations — implementations in separate .cpp files
at::Tensor rms_norm(at::Tensor x, at::Tensor weight, double eps);
at::Tensor apply_rope(at::Tensor x, at::Tensor cos_sin_cache, at::Tensor positions, bool is_neox_style);
at::Tensor fused_mla_linear(at::Tensor x, at::Tensor weight, c10::optional<at::Tensor> bias);
at::Tensor kv_b_proj_forward(at::Tensor x, at::Tensor weight, c10::optional<at::Tensor> bias);
at::Tensor concat_k_nope_k_pe(at::Tensor k_nope, at::Tensor k_pe);
std::tuple<at::Tensor, at::Tensor> build_absorption_matrices(at::Tensor kv_b_proj_weight, int64_t num_heads, int64_t qk_nope_head_dim, int64_t v_head_dim, int64_t kv_lora_rank, at::ScalarType dtype);
void write_kv_cache(at::Tensor kv_c, at::Tensor k_pe, at::Tensor kv_cache, at::Tensor slot_mapping);
at::Tensor gather_kv_cache(at::Tensor kv_cache, at::Tensor block_table, int64_t seq_len, int64_t block_size);
at::Tensor merge_attn_states(at::Tensor prefix_output, at::Tensor prefix_lse, at::Tensor suffix_output, at::Tensor suffix_lse);
py::object varlen_attention(at::Tensor q, at::Tensor k, at::Tensor v,
                            at::Tensor cu_seqlens_q, at::Tensor cu_seqlens_k,
                            int64_t max_seqlen_q, int64_t max_seqlen_k,
                            double scale, bool causal, bool return_softmax_lse);
at::Tensor forward_decode(at::Tensor q_nope_proj, at::Tensor q_pe,
                          at::Tensor kv_cache, at::Tensor block_table,
                          at::Tensor seq_lens, double scale,
                          int64_t kv_lora_rank, int64_t qk_rope_head_dim);
at::Tensor scaled_dot_product_attention(
    at::Tensor query, at::Tensor key, at::Tensor value,
    c10::optional<at::Tensor> attn_mask, double dropout_p,
    bool is_causal, c10::optional<double> scale, bool enable_gqa);
at::Tensor scaled_dot_product_attention_versioned(
    at::Tensor query, at::Tensor key, at::Tensor value,
    c10::optional<at::Tensor> attn_mask, double dropout_p,
    bool is_causal, c10::optional<double> scale, bool enable_gqa,
    std::string version);
at::Tensor multi_query_attention(
    at::Tensor query, at::Tensor key, at::Tensor value,
    c10::optional<at::Tensor> attn_mask, double dropout_p,
    bool is_causal, c10::optional<double> scale);
std::vector<std::string> list_sdpa_versions();
std::map<std::string, double> validate_sdpa_flash2_neon_cache_microkernels(
    std::string dtype, int64_t E, int64_t Sk);
std::map<std::string, double> benchmark_sdpa_flash2_neon_cache_microkernels(
    std::string dtype, int64_t E, int64_t Sk, int64_t iterations,
    int64_t warmup);

// 微内核管理框架 — 见 csrc/sdpa_microkernels/mk_registry.cpp
std::vector<std::string> list_microkernel_impls();
std::map<std::string, double> validate_microkernel(
    std::string impl, std::string dtype, int64_t E, int64_t Sk);
std::map<std::string, double> benchmark_microkernel(
    std::string impl, std::string dtype, int64_t E, int64_t Sk,
    int64_t iterations, int64_t warmup);

// 微内核管理框架 — 见 csrc/sdpa_microkernels/mk_registry.cpp
std::vector<std::string> list_microkernel_impls();
std::map<std::string, double> validate_microkernel(
    std::string impl, std::string dtype, int64_t E, int64_t Sk);
std::map<std::string, double> benchmark_microkernel(
    std::string impl, std::string dtype, int64_t E, int64_t Sk,
    int64_t iterations, int64_t warmup);
py::object flash_mla_sparse_fwd(
    at::Tensor q,
    at::Tensor kv,
    at::Tensor indices,
    double sm_scale,
    c10::optional<int64_t> d_v,
    c10::optional<at::Tensor> attn_sink,
    c10::optional<at::Tensor> topk_length,
    c10::optional<at::Tensor> out,
    bool return_stats);
at::Tensor sparse_attn_indexer_prefill_cpp_v0(
    at::Tensor q_quant,
    at::Tensor weights,
    at::Tensor kv_cache,
    at::Tensor topk_indices_buffer,
    int64_t topk_tokens,
    py::object attn_metadata);

// OMP runtime info forward declarations — omp_info.cpp
std::map<std::string, std::string> get_omp_runtime_info();
bool has_openmp();

// ACL GEMM forward declarations — acl_gemm.cpp
int64_t create_acl_gemm_handler(at::Tensor weight, int64_t num_threads,
                                bool fast_math);
void acl_gemm(at::Tensor output, at::Tensor input,
             c10::optional<at::Tensor> bias, int64_t handler_ptr);
void release_acl_gemm_handler(int64_t handler_ptr);

// KAI GEMM forward declarations — kai_gemm.cpp
at::Tensor kai_gemm_prepare(at::Tensor weight,
                            c10::optional<at::Tensor> bias);
int64_t create_kai_thread_pool(std::vector<int64_t> cpu_ids);
void destroy_kai_thread_pool(int64_t pool_handle);
int64_t create_kai_gemm_handler(at::Tensor packed_weight,
                                int64_t K, int64_t N);
void kai_gemm(at::Tensor output, at::Tensor input,
              int64_t handler_ptr, int64_t pool_handle);
void release_kai_gemm_handler(int64_t handler_ptr);

// I8 GEMM declarations - i8gemm.cpp
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t,
           std::string>
i8gemm_prepare(at::Tensor weight, at::Tensor weight_scale,
               std::string backend);
void i8gemm_dynamic_scaled_mm(at::Tensor output,
                              at::Tensor input,
                              at::Tensor packed_weight,
                              at::Tensor weight_scale,
                              c10::optional<at::Tensor> bias,
                              int64_t K,
                              int64_t N,
                              int64_t Kp,
                              int64_t Np,
                              int64_t nthreads);

// BF16 GEMM declarations - bf16_linear.cpp
std::tuple<at::Tensor, int64_t, int64_t, int64_t>
bf16_linear_prepare_weight(at::Tensor weight);
at::Tensor bf16_linear_prepacked_to_dtype(at::Tensor input,
                                          at::Tensor packed_weight,
                                          int64_t K,
                                          int64_t N,
                                          int64_t Np,
                                          bool output_bf16,
                                          int64_t nthreads);
at::Tensor bf16_linear_to_dtype(at::Tensor input,
                                at::Tensor weight,
                                bool output_bf16,
                                int64_t nthreads);

// BF16 tiled fused MoE declarations — fused_moe_bf16_tiled.cpp
std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor w13_weight,
                                      at::Tensor w2_weight);
at::Tensor fused_moe_bf16_tiled(at::Tensor input,
                                at::Tensor w13_packed,
                                int64_t w13_K,
                                int64_t w13_N,
                                at::Tensor w2_packed,
                                int64_t w2_K,
                                int64_t w2_N,
                                at::Tensor topk_weights,
                                at::Tensor topk_ids,
                                c10::optional<at::Tensor> w13_bias,
                                c10::optional<at::Tensor> w2_bias,
                                int64_t num_threads,
                                std::string activation,
                                int64_t global_num_experts,
                                bool skip_weighted);

// DeepSeek V4 attn_gemm_parallel_execute fused GEMM declarations
std::tuple<at::Tensor, int64_t, int64_t>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare(
    at::Tensor weight);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
    at::Tensor hidden_states,
    at::Tensor fused_wqa_wkv_packed,
    int64_t fused_wqa_wkv_K,
    int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed,
    int64_t compressor_kv_score_K,
    int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed,
    int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N,
    at::Tensor indexer_weights_proj_packed,
    int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt(
    at::Tensor hidden_states,
    at::Tensor fused_wqa_wkv_packed,
    int64_t fused_wqa_wkv_K,
    int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed,
    int64_t compressor_kv_score_K,
    int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed,
    int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N,
    at::Tensor indexer_weights_proj_packed,
    int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N,
    std::vector<int64_t> core_ids);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage(
    at::Tensor qr,
    at::Tensor kv,
    at::Tensor kv_score,
    at::Tensor indexer_kv_score,
    at::Tensor indexer_weights,
    at::Tensor positions,
    at::Tensor main_wq_b_weight,
    at::Tensor indexer_wq_b_weight,
    at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache,
    at::Tensor swa_kv_cache,
    at::Tensor swa_slot_mapping,
    at::Tensor mla_ape,
    at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping,
    at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table,
    at::Tensor mla_kv_cache,
    at::Tensor mla_kv_slot_mapping,
    at::Tensor mla_norm_weight,
    at::Tensor indexer_ape,
    at::Tensor indexer_state_cache,
    at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices,
    at::Tensor indexer_block_table,
    at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping,
    at::Tensor indexer_norm_weight,
    at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens,
    at::Tensor prefill_cu_seqlen_ks,
    at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table,
    int64_t main_head_dim,
    double q_eps,
    int64_t mla_compress_ratio,
    double mla_rms_norm_eps,
    int64_t indexer_compress_ratio,
    double indexer_rms_norm_eps,
    int64_t topk_tokens);
std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage_prepacked(
    at::Tensor qr,
    at::Tensor kv,
    at::Tensor kv_score,
    at::Tensor indexer_kv_score,
    at::Tensor indexer_weights,
    at::Tensor positions,
    at::Tensor main_wq_b_packed,
    int64_t main_wq_b_K,
    int64_t main_wq_b_N,
    int64_t main_wq_b_Np,
    at::Tensor indexer_wq_b_packed,
    int64_t indexer_wq_b_K,
    int64_t indexer_wq_b_N,
    int64_t indexer_wq_b_Np,
    at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache,
    at::Tensor swa_kv_cache,
    at::Tensor swa_slot_mapping,
    at::Tensor mla_ape,
    at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping,
    at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table,
    at::Tensor mla_kv_cache,
    at::Tensor mla_kv_slot_mapping,
    at::Tensor mla_norm_weight,
    at::Tensor indexer_ape,
    at::Tensor indexer_state_cache,
    at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices,
    at::Tensor indexer_block_table,
    at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping,
    at::Tensor indexer_norm_weight,
    at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens,
    at::Tensor prefill_cu_seqlen_ks,
    at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table,
    int64_t main_head_dim,
    double q_eps,
    int64_t mla_compress_ratio,
    double mla_rms_norm_eps,
    int64_t indexer_compress_ratio,
    double indexer_rms_norm_eps,
    int64_t topk_tokens);

// ACL affinity forward declarations — acl_affinity.cpp
void set_acl_thread_affinity(int64_t core_start, int64_t core_end,
                             int64_t num_threads);
std::tuple<int64_t, int64_t, int64_t> get_acl_thread_affinity();

PYBIND11_MODULE(_C, m) {
    m.doc() = "fused_cpp C++ extension kernels";

    m.def("rms_norm", &rms_norm, "RMSNorm: normalize in fp32, cast back, multiply by weight",
          py::arg("x"), py::arg("weight"), py::arg("eps"));

    m.def("apply_rope", &apply_rope, "Rotary position embedding (NeoX or GPT-J)",
          py::arg("x"), py::arg("cos_sin_cache"), py::arg("positions"), py::arg("is_neox_style"));

    m.def("linear", &fused_mla_linear, "F.linear equivalent",
          py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt);

    m.def("kv_b_proj_forward", &kv_b_proj_forward, "kv_b_proj linear transform",
          py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt);

    m.def("concat_k_nope_k_pe", &concat_k_nope_k_pe, "Broadcast k_pe to num_heads, concatenate",
          py::arg("k_nope"), py::arg("k_pe"));

    m.def("build_absorption_matrices", &build_absorption_matrices,
          "Build W_UK_T and W_UV from kv_b_proj weight",
          py::arg("kv_b_proj_weight"), py::arg("num_heads"),
          py::arg("qk_nope_head_dim"), py::arg("v_head_dim"),
          py::arg("kv_lora_rank"), py::arg("dtype"));

    m.def("write_kv_cache", &write_kv_cache,
          "Write concatenated kv_c+k_pe to paged KV cache via slot_mapping",
          py::arg("kv_c"), py::arg("k_pe"), py::arg("kv_cache"), py::arg("slot_mapping"));

    m.def("gather_kv_cache", &gather_kv_cache,
          "Gather seq_len tokens from paged KV cache",
          py::arg("kv_cache"), py::arg("block_table"), py::arg("seq_len"), py::arg("block_size"));

    m.def("merge_attn_states", &merge_attn_states,
          "LSE merge of two attention outputs",
          py::arg("prefix_output"), py::arg("prefix_lse"),
          py::arg("suffix_output"), py::arg("suffix_lse"));

    m.def("varlen_attention", &varlen_attention,
          "Variable-length multi-head attention with optional LSE return",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("scale"), py::arg("causal"), py::arg("return_softmax_lse"));

    m.def("forward_decode", &forward_decode,
          "Decode-stage MQA absorption attention",
          py::arg("q_nope_proj"), py::arg("q_pe"),
          py::arg("kv_cache"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("scale"),
          py::arg("kv_lora_rank"), py::arg("qk_rope_head_dim"));

    m.def("scaled_dot_product_attention", &scaled_dot_product_attention,
          "Scaled dot-product attention (SDPA) with float32 accumulation",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("attn_mask") = c10::nullopt,
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = c10::nullopt,
          py::arg("enable_gqa") = false);

    m.def("scaled_dot_product_attention_versioned",
          &scaled_dot_product_attention_versioned,
          "Versioned SDPA: dispatch to a registered kernel by name",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("attn_mask") = c10::nullopt,
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = c10::nullopt,
          py::arg("enable_gqa") = false,
          py::arg("version"));

    m.def("multi_query_attention", &multi_query_attention,
          "Dense MQA attention with shared single-head K/V",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("attn_mask") = c10::nullopt,
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = c10::nullopt,
          py::call_guard<py::gil_scoped_release>());

    m.def("list_sdpa_versions", &list_sdpa_versions,
          "Return the list of registered SDPA version names");

    m.def("validate_sdpa_flash2_neon_cache_microkernels",
          &validate_sdpa_flash2_neon_cache_microkernels,
          "Validate flash2_neon_cache QKT/PV micro-kernels against scalar references.",
          py::arg("dtype") = "bf16",
          py::arg("E") = 128,
          py::arg("Sk") = 128,
          py::call_guard<py::gil_scoped_release>());

    m.def("benchmark_sdpa_flash2_neon_cache_microkernels",
          &benchmark_sdpa_flash2_neon_cache_microkernels,
          "Benchmark flash2_neon_cache QKT/PV micro-kernels and return timing/GFLOPS.",
          py::arg("dtype") = "bf16",
          py::arg("E") = 128,
          py::arg("Sk") = 128,
          py::arg("iterations") = 100000,
          py::arg("warmup") = 1000,
          py::call_guard<py::gil_scoped_release>());

    // ── 微内核管理框架 ────────────────────────────────────────────────
    m.def("list_microkernel_impls", &list_microkernel_impls,
          "List all microkernel impls registered by the management framework "
          "(name strings; one per FUSED_CPP_MK_ENABLE_<NAME> compile-time flag).");

    m.def("validate_microkernel", &validate_microkernel,
          "Validate a microkernel impl against its scalar reference; returns "
          "max-abs error per op (qkt_8x8 / qkt_8x4 / qkt_tail / pv_8x8 / pv_tail).",
          py::arg("impl"), py::arg("dtype") = "bf16",
          py::arg("E") = 128, py::arg("Sk") = 128,
          py::call_guard<py::gil_scoped_release>());

    m.def("benchmark_microkernel", &benchmark_microkernel,
          "Benchmark a microkernel impl; returns seconds / us / GFLOPS / "
          "checksum per op (qkt_8x8 / qkt_8x4 / pv_8x8).",
          py::arg("impl"), py::arg("dtype") = "bf16",
          py::arg("E") = 128, py::arg("Sk") = 128,
          py::arg("iterations") = 100000, py::arg("warmup") = 1000,
          py::call_guard<py::gil_scoped_release>());

    m.def("flash_mla_sparse_fwd", &flash_mla_sparse_fwd,
          "Plan-based sparse MLA forward. Dense shared KV runs reuse 8x8 "
          "NEON SDPA microkernels; indexed tails use fp32 FMLA/scalar fallback.",
          py::arg("q"),
          py::arg("kv"),
          py::arg("indices"),
          py::arg("sm_scale"),
          py::arg("d_v") = c10::nullopt,
          py::arg("attn_sink") = c10::nullopt,
          py::arg("topk_length") = c10::nullopt,
          py::arg("out") = c10::nullopt,
          py::arg("return_stats") = false,
          py::call_guard<py::gil_scoped_release>());

    m.def("sparse_attn_indexer_prefill_cpp_v0",
          &sparse_attn_indexer_prefill_cpp_v0,
          "Deprecated DeepSeek V4 sparse attention indexer prefill implementation. "
          "This symbol is retained only to raise a clear error; use the "
          "Python/Torch sparse indexer baseline or deepseek_v4_post_gemm_parallel_stage.",
          py::arg("q_quant"),
          py::arg("weights"),
          py::arg("kv_cache"),
          py::arg("topk_indices_buffer"),
          py::arg("topk_tokens"),
          py::arg("attn_metadata"));

    m.def("get_omp_runtime_info", &get_omp_runtime_info,
          "Return a snapshot of the current OpenMP runtime configuration "
          "(env vars + omp_get_max_threads / num_procs).");

    m.def("has_openmp", &has_openmp,
          "Return True if the C++ extension was linked with OpenMP.");

    // ── I8 GEMM 接口 ──
    m.def("i8gemm_prepare", &i8gemm_prepare,
          "Prepare int8 [N, K] vLLM linear weight for i8 GEMM.",
          py::arg("weight"), py::arg("weight_scale"),
          py::arg("backend") = "auto",
          py::call_guard<py::gil_scoped_release>());

    m.def("i8gemm_dynamic_scaled_mm", &i8gemm_dynamic_scaled_mm,
          "Dynamic per-token scaled int8 GEMM with fp32/bf16 output.",
          py::arg("output"), py::arg("input"),
          py::arg("packed_weight"), py::arg("weight_scale"),
          py::arg("bias") = c10::nullopt,
          py::arg("K"), py::arg("N"), py::arg("Kp"), py::arg("Np"),
          py::arg("nthreads") = 0,
          py::call_guard<py::gil_scoped_release>());

    m.def("bf16_linear_to_dtype", &bf16_linear_to_dtype,
          "BF16 [M,K] x [N,K]^T linear using refs/i8gemm bf16gemm. "
          "The accumulator/output kernel is fp32; output_bf16 casts the "
          "visible result to bf16.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output_bf16"),
          py::arg("nthreads") = 0,
          py::call_guard<py::gil_scoped_release>());

    m.def("bf16_linear_prepare_weight", &bf16_linear_prepare_weight,
          "Prepack a bf16 PyTorch linear weight [N,K] for refs/i8gemm "
          "bf16gemm. Returns (packed_weight, K, N, Np).",
          py::arg("weight"),
          py::call_guard<py::gil_scoped_release>());

    m.def("bf16_linear_prepacked_to_dtype", &bf16_linear_prepacked_to_dtype,
          "BF16 [M,K] linear using a prepacked refs/i8gemm bf16gemm weight.",
          py::arg("input"),
          py::arg("packed_weight"),
          py::arg("K"),
          py::arg("N"),
          py::arg("Np"),
          py::arg("output_bf16"),
          py::arg("nthreads") = 0,
          py::call_guard<py::gil_scoped_release>());

    m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare",
          &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare,
          "Pack one bf16 [K, N] weight for the DeepSeek V4 4-GEMM fused path.",
          py::arg("weight"),
          py::call_guard<py::gil_scoped_release>());

    m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused",
          &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused,
          "Serial DeepSeek V4 attn_gemm_parallel_execute fused path: "
          "fused_wqa_wkv bf16, compressor_kv_score fp32, "
          "indexer_compressor_kv_score fp32, indexer_weights_proj bf16.",
          py::arg("hidden_states"),
          py::arg("fused_wqa_wkv_packed"),
          py::arg("fused_wqa_wkv_K"),
          py::arg("fused_wqa_wkv_N"),
          py::arg("compressor_kv_score_packed"),
          py::arg("compressor_kv_score_K"),
          py::arg("compressor_kv_score_N"),
          py::arg("indexer_compressor_kv_score_packed"),
          py::arg("indexer_compressor_kv_score_K"),
          py::arg("indexer_compressor_kv_score_N"),
          py::arg("indexer_weights_proj_packed"),
          py::arg("indexer_weights_proj_K"),
          py::arg("indexer_weights_proj_N"),
          py::call_guard<py::gil_scoped_release>());

    m.def("fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt",
          &fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt,
          "OpenMP DeepSeek V4 attn_gemm_parallel_execute fused path. "
          "core_ids controls thread count and per-thread CPU affinity; "
          "rows are split into contiguous ceil(M / len(core_ids)) chunks.",
          py::arg("hidden_states"),
          py::arg("fused_wqa_wkv_packed"),
          py::arg("fused_wqa_wkv_K"),
          py::arg("fused_wqa_wkv_N"),
          py::arg("compressor_kv_score_packed"),
          py::arg("compressor_kv_score_K"),
          py::arg("compressor_kv_score_N"),
          py::arg("indexer_compressor_kv_score_packed"),
          py::arg("indexer_compressor_kv_score_K"),
          py::arg("indexer_compressor_kv_score_N"),
          py::arg("indexer_weights_proj_packed"),
          py::arg("indexer_weights_proj_K"),
          py::arg("indexer_weights_proj_N"),
          py::arg("core_ids"),
          py::call_guard<py::gil_scoped_release>());

    m.def("deepseek_v4_post_gemm_parallel_stage",
          &deepseek_v4_post_gemm_parallel_stage,
          "Torch-API C++ baseline for the DeepSeek V4 post-GEMM attention "
          "stage. Covers the CPU prefill semantics of the second "
          "execute_in_parallel block: main q path, MLA compressor, indexer "
          "compressor, and sparse indexer top-k.",
          py::arg("qr"),
          py::arg("kv"),
          py::arg("kv_score"),
          py::arg("indexer_kv_score"),
          py::arg("indexer_weights"),
          py::arg("positions"),
          py::arg("main_wq_b_weight"),
          py::arg("indexer_wq_b_weight"),
          py::arg("main_cos_sin_cache"),
          py::arg("indexer_cos_sin_cache"),
          py::arg("swa_kv_cache"),
          py::arg("swa_slot_mapping"),
          py::arg("mla_ape"),
          py::arg("mla_state_cache"),
          py::arg("mla_state_slot_mapping"),
          py::arg("mla_token_to_req_indices"),
          py::arg("mla_block_table"),
          py::arg("mla_kv_cache"),
          py::arg("mla_kv_slot_mapping"),
          py::arg("mla_norm_weight"),
          py::arg("indexer_ape"),
          py::arg("indexer_state_cache"),
          py::arg("indexer_state_slot_mapping"),
          py::arg("indexer_token_to_req_indices"),
          py::arg("indexer_block_table"),
          py::arg("indexer_kv_cache"),
          py::arg("indexer_kv_slot_mapping"),
          py::arg("indexer_norm_weight"),
          py::arg("topk_indices_buffer"),
          py::arg("prefill_cu_seq_lens"),
          py::arg("prefill_cu_seqlen_ks"),
          py::arg("prefill_cu_seqlen_ke"),
          py::arg("prefill_block_table"),
          py::arg("main_head_dim"),
          py::arg("q_eps"),
          py::arg("mla_compress_ratio"),
          py::arg("mla_rms_norm_eps"),
          py::arg("indexer_compress_ratio"),
          py::arg("indexer_rms_norm_eps"),
          py::arg("topk_tokens"),
          py::call_guard<py::gil_scoped_release>());

    m.def("deepseek_v4_post_gemm_parallel_stage_prepacked",
          &deepseek_v4_post_gemm_parallel_stage_prepacked,
          "Torch-API C++ baseline for the DeepSeek V4 post-GEMM attention "
          "stage using prepacked bf16 weights for the two post linear layers.",
          py::arg("qr"),
          py::arg("kv"),
          py::arg("kv_score"),
          py::arg("indexer_kv_score"),
          py::arg("indexer_weights"),
          py::arg("positions"),
          py::arg("main_wq_b_packed"),
          py::arg("main_wq_b_K"),
          py::arg("main_wq_b_N"),
          py::arg("main_wq_b_Np"),
          py::arg("indexer_wq_b_packed"),
          py::arg("indexer_wq_b_K"),
          py::arg("indexer_wq_b_N"),
          py::arg("indexer_wq_b_Np"),
          py::arg("main_cos_sin_cache"),
          py::arg("indexer_cos_sin_cache"),
          py::arg("swa_kv_cache"),
          py::arg("swa_slot_mapping"),
          py::arg("mla_ape"),
          py::arg("mla_state_cache"),
          py::arg("mla_state_slot_mapping"),
          py::arg("mla_token_to_req_indices"),
          py::arg("mla_block_table"),
          py::arg("mla_kv_cache"),
          py::arg("mla_kv_slot_mapping"),
          py::arg("mla_norm_weight"),
          py::arg("indexer_ape"),
          py::arg("indexer_state_cache"),
          py::arg("indexer_state_slot_mapping"),
          py::arg("indexer_token_to_req_indices"),
          py::arg("indexer_block_table"),
          py::arg("indexer_kv_cache"),
          py::arg("indexer_kv_slot_mapping"),
          py::arg("indexer_norm_weight"),
          py::arg("topk_indices_buffer"),
          py::arg("prefill_cu_seq_lens"),
          py::arg("prefill_cu_seqlen_ks"),
          py::arg("prefill_cu_seqlen_ke"),
          py::arg("prefill_block_table"),
          py::arg("main_head_dim"),
          py::arg("q_eps"),
          py::arg("mla_compress_ratio"),
          py::arg("mla_rms_norm_eps"),
          py::arg("indexer_compress_ratio"),
          py::arg("indexer_rms_norm_eps"),
          py::arg("topk_tokens"),
          py::call_guard<py::gil_scoped_release>());

    m.def("deepseek_v4_q_norm_rope_fused_sve",
          &::fused_cpp::deepseek_v4::q_norm_rope_fused_sve,
          "Run only the DeepSeek V4 q RMSNorm+RoPE SVE fast path in-place. "
          "Inputs must already be CPU tensors with q stride(-1)=1, positions "
          "as int64, and cos_sin_cache as contiguous float32.",
          py::arg("q"),
          py::arg("positions_long"),
          py::arg("cos_sin_f"),
          py::arg("eps"),
          py::call_guard<py::gil_scoped_release>());

    // ── ACL GEMM 接口 ──
#if defined(__aarch64__) && defined(FUSED_CPP_HAS_ACL)
    m.def("create_acl_gemm_handler", &create_acl_gemm_handler,
          "Create ACL GEMM handler with weight prepacking",
          py::arg("weight"), py::arg("num_threads") = 0,
          py::arg("fast_math") = false);

    m.def("acl_gemm", &acl_gemm,
          "Execute GEMM using ACL with prepacked weights",
          py::arg("output"), py::arg("input"),
          py::arg("bias"), py::arg("handler_ptr"));

    m.def("release_acl_gemm_handler", &release_acl_gemm_handler,
          "Release ACL GEMM handler and free resources",
          py::arg("handler_ptr"));

    m.def("set_acl_thread_affinity", &set_acl_thread_affinity,
          "Set ACL thread affinity and core binding",
          py::arg("core_start"), py::arg("core_end"),
          py::arg("num_threads"));

    m.def("get_acl_thread_affinity", &get_acl_thread_affinity,
          "Get current ACL thread affinity configuration");
#endif

    // ── KAI GEMM 接口 ──
#ifdef __aarch64__
    m.def("kai_gemm_prepare", &kai_gemm_prepare,
          "Prepare (prepack) weight and optional bias for KAI GEMM",
          py::arg("weight"), py::arg("bias") = c10::nullopt);

    m.def("create_kai_thread_pool", &create_kai_thread_pool,
          "Create a KAI thread pool bound to given CPU ids; "
          "returns an int64 handle owned by the native registry.",
          py::arg("cpu_ids"));

    m.def("destroy_kai_thread_pool", &destroy_kai_thread_pool,
          "Destroy a KAI thread pool previously returned by "
          "create_kai_thread_pool; 0/invalid handles are ignored.",
          py::arg("pool_handle"));

    m.def("create_kai_gemm_handler", &create_kai_gemm_handler,
          "Create KAI GEMM handler from packed weight (pure data handler, "
          "no thread resource attached)",
          py::arg("packed_weight"), py::arg("K"), py::arg("N"));

    m.def("kai_gemm", &kai_gemm,
          "Execute GEMM using KleidiAI microkernels; set pool_handle=0 "
          "for the single-threaded path.",
          py::arg("output"), py::arg("input"), py::arg("handler_ptr"),
          py::arg("pool_handle") = 0,
          py::call_guard<py::gil_scoped_release>());

    m.def("release_kai_gemm_handler", &release_kai_gemm_handler,
          "Release KAI GEMM handler and free resources",
          py::arg("handler_ptr"));
#endif

    // ── BF16 tiled fused MoE ──
#ifdef __aarch64__
    m.def("fused_moe_bf16_tiled_prepare_weights",
          &fused_moe_bf16_tiled_prepare_weights,
          "Pack BF16 MoE expert weights for the BF16 tiled fused MoE path.",
          py::arg("w13_weight"), py::arg("w2_weight"),
          py::call_guard<py::gil_scoped_release>());

    m.def("fused_moe_bf16_tiled",
          &fused_moe_bf16_tiled,
          "Run BF16 tiled fused MoE. Expert route groups are split "
          "into <=16-token tiles and assigned to a fixed thread count by "
          "routed-token load.",
          py::arg("input"),
          py::arg("w13_packed"),
          py::arg("w13_K"),
          py::arg("w13_N"),
          py::arg("w2_packed"),
          py::arg("w2_K"),
          py::arg("w2_N"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("w13_bias") = c10::nullopt,
          py::arg("w2_bias") = c10::nullopt,
          py::arg("num_threads") = 1,
          py::arg("activation") = "silu",
          py::arg("global_num_experts") = -1,
          py::arg("skip_weighted") = false,
          py::call_guard<py::gil_scoped_release>());
#endif
}
