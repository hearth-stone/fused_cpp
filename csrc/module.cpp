#include <torch/extension.h>

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

PYBIND11_MODULE(_C, m) {
    m.doc() = "fused_mla_cpp C++ extension kernels";

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
}
