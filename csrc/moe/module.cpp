// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <pybind11/stl.h>

#include "common/api.h"
#include "common/backend.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  fused_cpp::moe::validate_sve_vector_length_at_import();

  m.def("fused_moe_bf16_tiled_available_backends", &fused_cpp::moe::available_backend_names,
        "Return the BF16 fused MoE backends supported by this build and CPU.");

  m.def("fused_moe_bf16_tiled_prepare_weights", &fused_moe_bf16_tiled_prepare_weights,
        "Pack BF16 MoE expert weights for the selected ISA backend.", py::arg("w13_weight"), py::arg("w2_weight"),
        py::arg("fuse_silu") = false, py::arg("backend") = "auto", py::call_guard<py::gil_scoped_release>());

#if defined(__aarch64__)
  m.def("fused_moe_test_split_plan", &fused_moe_test_split_plan, "Test-only: return the cooperative GEMM split plan.",
        py::arg("stage"), py::arg("M"), py::arg("K"), py::arg("N"), py::arg("group_size"),
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_single_thread_gemm", &fused_moe_test_single_thread_gemm, "Test-only: run one BF16 GEMM.",
        py::arg("A"), py::arg("B"), py::arg("bias") = c10::nullopt, py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_pack_interleaved_gemm", &fused_moe_test_pack_interleaved_gemm,
        "Test-only: pack interleaved w13 and run GEMM.", py::arg("A"), py::arg("w13"),
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_fused_w13_linear", &fused_moe_test_fused_w13_linear,
        "Test-only: run the fused w13 gate/up kernel.", py::arg("A"), py::arg("w13"),
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_fused_w13_silu", &fused_moe_test_fused_w13_silu, "Test-only: run fused w13 SiLU-and-mul.",
        py::arg("A"), py::arg("w13"), py::arg("degree") = 5, py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_team_fused_w13_silu", &fused_moe_test_team_fused_w13_silu,
        "Test-only: run cooperative fused w13 SiLU-and-mul.", py::arg("A"), py::arg("w13"), py::arg("group_size"),
        py::arg("degree") = 5, py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_pack_a_reorder_m8", &fused_moe_test_pack_a_reorder_m8,
        "Test-only: pack A in reorder-m8 layout.", py::arg("A"), py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_gather_pack_a_reorder_m8", &fused_moe_test_gather_pack_a_reorder_m8,
        "Test-only: fuse route gather with reorder-m8 packing.", py::arg("input"), py::arg("routes"), py::arg("top_k"),
        py::arg("K_pad"), py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_fused_w13_silu_packc", &fused_moe_test_fused_w13_silu_packc,
        "Test-only: run fused w13 SiLU-and-mul with packed-C store.", py::arg("A"), py::arg("w13"),
        py::arg("degree") = 5, py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_fused_w13_silu_packc_tail", &fused_moe_test_fused_w13_silu_packc_tail,
        "Test-only: run tail-aware fused w13 packed-C.", py::arg("A"), py::arg("w13"), py::arg("degree") = 5,
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_bench_fused_w13_silu_packc_tail", &fused_moe_bench_fused_w13_silu_packc_tail,
        "Benchmark fused w13 packed-C tail dispatch.", py::arg("A"), py::arg("w13"), py::arg("degree") = 5,
        py::arg("mode") = 0, py::arg("warmup") = 20, py::arg("runs") = 100, py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_team_gemm", &fused_moe_test_team_gemm, "Test-only: run cooperative team GEMM.", py::arg("A"),
        py::arg("B"), py::arg("group_size"), py::arg("split") = "auto", py::arg("bias") = c10::nullopt,
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_bench_team_gemm", &fused_moe_bench_team_gemm, "Benchmark cooperative team GEMM.", py::arg("A"),
        py::arg("B"), py::arg("group_size"), py::arg("split") = "auto", py::arg("bias") = c10::nullopt,
        py::arg("warmup") = 3, py::arg("runs") = 20, py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_test_sve_packed_gemm", &fused_moe_test_sve_packed_gemm,
        "Test-only: run standalone packed-A/B SVE GEMM with JIT or static asm.", py::arg("A"), py::arg("packed_B"),
        py::arg("K"), py::arg("N"), py::arg("n_tile"), py::arg("use_jit") = true,
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_moe_bench_sve_jit_w13_gemm", &fused_moe_bench_sve_jit_w13_gemm,
        "Benchmark only the exact-M SVE JIT GEMM body on packed W13 weights.", py::arg("A"),
        py::arg("w13_packed"), py::arg("K"), py::arg("N"), py::arg("n_tile"), py::arg("n_ranges") = 2,
        py::arg("warmup") = 64, py::arg("runs") = 192, py::arg("probe_mode") = 0,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def("fused_moe_bf16_tiled", &fused_moe_bf16_tiled, "Run BF16 tiled fused MoE.", py::arg("input"),
        py::arg("w13_packed"), py::arg("w13_K"), py::arg("w13_N"), py::arg("w2_packed"), py::arg("w2_K"),
        py::arg("w2_N"), py::arg("topk_weights"), py::arg("topk_ids"), py::arg("w13_bias") = c10::nullopt,
        py::arg("w2_bias") = c10::nullopt, py::arg("num_threads") = 1, py::arg("activation") = "silu",
        py::arg("global_num_experts") = -1, py::arg("skip_weighted") = false, py::arg("fuse_silu") = false,
        py::arg("silu_poly_degree") = 5, py::arg("gemm_backend") = 0, py::arg("backend_n_tile") = 8,
        py::arg("weight_window_bytes") = -1, py::arg("out") = c10::nullopt, py::call_guard<py::gil_scoped_release>());

  m.def("fused_moe_bf16_tiled_scheduled", &fused_moe_bf16_tiled_scheduled,
        "Run BF16 tiled fused MoE with a wave/team schedule.", py::arg("input"), py::arg("w13_packed"),
        py::arg("w13_K"), py::arg("w13_N"), py::arg("w2_packed"), py::arg("w2_K"), py::arg("w2_N"),
        py::arg("topk_weights"), py::arg("topk_ids"), py::arg("wave_offsets"), py::arg("team_expert_ids"),
        py::arg("team_threads"), py::arg("thread_cpu_ids") = c10::nullopt, py::arg("w13_bias") = c10::nullopt,
        py::arg("w2_bias") = c10::nullopt, py::arg("num_threads") = 1, py::arg("activation") = "silu",
        py::arg("global_num_experts") = -1, py::arg("skip_weighted") = false, py::arg("fuse_silu") = false,
        py::arg("silu_poly_degree") = 5, py::arg("gemm_backend") = 0, py::arg("backend_n_tile") = 8,
        py::arg("weight_window_bytes") = -1, py::arg("out") = c10::nullopt, py::call_guard<py::gil_scoped_release>());

  m.def("fused_moe_bf16_tiled_async", &fused_moe_bf16_tiled_async, "Run BF16 tiled fused MoE with an async task DAG.",
        py::arg("input"), py::arg("w13_packed"), py::arg("w13_K"), py::arg("w13_N"), py::arg("w2_packed"),
        py::arg("w2_K"), py::arg("w2_N"), py::arg("topk_weights"), py::arg("topk_ids"), py::arg("task_expert_ids"),
        py::arg("task_core_begins"), py::arg("task_threads"), py::arg("task_dep_offsets"), py::arg("task_deps"),
        py::arg("thread_cpu_ids") = c10::nullopt, py::arg("w13_bias") = c10::nullopt, py::arg("w2_bias") = c10::nullopt,
        py::arg("num_threads") = 1, py::arg("activation") = "silu", py::arg("global_num_experts") = -1,
        py::arg("skip_weighted") = false, py::arg("fuse_silu") = false, py::arg("silu_poly_degree") = 5,
        py::arg("gemm_backend") = 0, py::arg("backend_n_tile") = 8, py::arg("w13_split") = -1,
        py::arg("weight_window_bytes") = -1, py::arg("out") = c10::nullopt, py::call_guard<py::gil_scoped_release>());

  m.def(
      "fused_moe_bf16_tiled_async_plan_v2", &fused_moe_bf16_tiled_async_plan_v2,
      "Run BF16 tiled fused MoE with a validated Plan V2 task DAG.", py::arg("input"), py::arg("w13_packed"),
      py::arg("w13_K"), py::arg("w13_N"), py::arg("w2_packed"), py::arg("w2_K"), py::arg("w2_N"),
      py::arg("topk_weights"), py::arg("topk_ids"), py::arg("task_expert_ids"), py::arg("task_core_begins"),
      py::arg("task_threads"), py::arg("task_dep_offsets"), py::arg("task_deps"), py::arg("plan_version"),
      py::arg("execution_mode"), py::arg("task_preferred_threads"), py::arg("task_min_threads"),
      py::arg("task_max_threads"), py::arg("task_allowed_thread_offsets"), py::arg("task_allowed_threads"),
      py::arg("task_placement_modes"), py::arg("task_numa_nodes"), py::arg("task_stage_ids"),
      py::arg("task_resize_points"), py::arg("task_range_granularities"),
      py::arg("thread_cpu_ids") = c10::nullopt, py::arg("w13_bias") = c10::nullopt,
      py::arg("w2_bias") = c10::nullopt, py::arg("num_threads") = 1, py::arg("activation") = "silu",
      py::arg("global_num_experts") = -1, py::arg("skip_weighted") = false, py::arg("fuse_silu") = false,
      py::arg("silu_poly_degree") = 5, py::arg("gemm_backend") = 0, py::arg("backend_n_tile") = 8,
      py::arg("w13_split") = -1, py::arg("weight_window_bytes") = -1, py::arg("out") = c10::nullopt,
      py::arg("task_w13_window_bytes") = c10::nullopt, py::arg("task_w2_window_bytes") = c10::nullopt,
      py::call_guard<py::gil_scoped_release>());

  m.def("fused_moe_bf16_tiled_vllm_staged", &fused_moe_bf16_tiled_vllm_staged,
        "Run the experimental vLLM-style global W13/W2 staged SVE baseline.", py::arg("input"),
        py::arg("w13_packed"), py::arg("w13_K"), py::arg("w13_N"), py::arg("w2_packed"), py::arg("w2_K"),
        py::arg("w2_N"), py::arg("topk_weights"), py::arg("topk_ids"), py::arg("thread_cpu_ids") = c10::nullopt,
        py::arg("num_threads") = 1, py::arg("global_num_experts") = -1, py::arg("fuse_silu") = true,
        py::arg("silu_poly_degree") = 5, py::arg("gemm_backend") = 1, py::arg("backend_n_tile") = 8,
        py::arg("out") = c10::nullopt, py::call_guard<py::gil_scoped_release>());
}
