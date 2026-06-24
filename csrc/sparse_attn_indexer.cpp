#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

constexpr int64_t kMaxCustomTopK = 64;

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

struct SparseAttnIndexerProfile {
  double input_check_ms = 0.0;
  double fill_ms = 0.0;
  double metadata_ms = 0.0;
  double short_path_ms = 0.0;
  double gather_ms = 0.0;
  double fold_q_ms = 0.0;
  double matmul_ms = 0.0;
  double topk_ms = 0.0;
  int64_t chunks = 0;
  int64_t short_chunks = 0;
  int64_t scored_chunks = 0;
};

bool ProfileEnabled() {
  const char* value = std::getenv("FUSED_CPP_SPARSE_ATTN_INDEXER_PROFILE");
  if (value == nullptr) {
    return false;
  }
  const std::string_view text(value);
  return !(text.empty() || text == "0" || text == "false" || text == "FALSE");
}

double ElapsedMs(TimePoint start) {
  const auto elapsed = Clock::now() - start;
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

void AddElapsed(double* field, TimePoint start) {
  *field += ElapsedMs(start);
}

void PrintProfile(const SparseAttnIndexerProfile& profile, double total_ms) {
  const double known_ms = profile.input_check_ms + profile.fill_ms + profile.metadata_ms + profile.short_path_ms +
      profile.gather_ms + profile.fold_q_ms + profile.matmul_ms + profile.topk_ms;
  const double other_ms = std::max(0.0, total_ms - known_ms);
  const auto pct = [total_ms](double ms) -> double {
    return total_ms > 0.0 ? (100.0 * ms / total_ms) : 0.0;
  };

  std::cerr << std::fixed << std::setprecision(3)
            << "sparse_attn_indexer_cpp_v0_profile"
            << " total_ms=" << total_ms
            << " input_check_ms=" << profile.input_check_ms << "(" << pct(profile.input_check_ms) << "%)"
            << " fill_ms=" << profile.fill_ms << "(" << pct(profile.fill_ms) << "%)"
            << " metadata_ms=" << profile.metadata_ms << "(" << pct(profile.metadata_ms) << "%)"
            << " short_path_ms=" << profile.short_path_ms << "(" << pct(profile.short_path_ms) << "%)"
            << " gather_ms=" << profile.gather_ms << "(" << pct(profile.gather_ms) << "%)"
            << " fold_q_ms=" << profile.fold_q_ms << "(" << pct(profile.fold_q_ms) << "%)"
            << " matmul_ms=" << profile.matmul_ms << "(" << pct(profile.matmul_ms) << "%)"
            << " topk_ms=" << profile.topk_ms << "(" << pct(profile.topk_ms) << "%)"
            << " other_ms=" << other_ms << "(" << pct(other_ms) << "%)"
            << " chunks=" << profile.chunks
            << " short_chunks=" << profile.short_chunks
            << " scored_chunks=" << profile.scored_chunks << '\n';
}

int64_t IntAttr(const py::handle& object, const char* name) {
  return py::cast<int64_t>(object.attr(name));
}

at::Tensor TensorAttr(const py::handle& object, const char* name) {
  return py::cast<at::Tensor>(object.attr(name));
}

void CheckSparseAttnIndexerInputs(
    const at::Tensor& q_quant,
    const at::Tensor& weights,
    const at::Tensor& kv_cache,
    const at::Tensor& topk_indices_buffer,
    int64_t topk_tokens) {
  TORCH_CHECK(q_quant.device().is_cpu(), "sparse_attn_indexer cpp_v0 requires q_quant on CPU");
  TORCH_CHECK(weights.device().is_cpu(), "sparse_attn_indexer cpp_v0 requires weights on CPU");
  TORCH_CHECK(kv_cache.device().is_cpu(), "sparse_attn_indexer cpp_v0 requires kv_cache on CPU");
  TORCH_CHECK(
      topk_indices_buffer.device().is_cpu(), "sparse_attn_indexer cpp_v0 requires topk_indices_buffer on CPU");
  TORCH_CHECK(q_quant.dim() == 3, "q_quant must be [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [num_tokens, num_heads]");
  TORCH_CHECK(kv_cache.dim() == 3, "kv_cache must be [num_blocks, block_size, head_dim]");
  TORCH_CHECK(topk_indices_buffer.dim() == 2, "topk_indices_buffer must be [max_tokens, max_topk]");
  TORCH_CHECK(q_quant.size(0) == weights.size(0), "q_quant/weights token dim mismatch");
  TORCH_CHECK(q_quant.size(1) == weights.size(1), "q_quant/weights head dim mismatch");
  TORCH_CHECK(q_quant.size(2) == kv_cache.size(2), "q_quant/kv_cache head_dim mismatch");
  TORCH_CHECK(topk_tokens >= 0, "topk_tokens must be non-negative");
  TORCH_CHECK(
      topk_tokens <= topk_indices_buffer.size(1),
      "topk_tokens exceeds topk_indices_buffer width: ",
      topk_tokens,
      " > ",
      topk_indices_buffer.size(1));
  TORCH_CHECK(
      topk_indices_buffer.scalar_type() == at::kInt || topk_indices_buffer.scalar_type() == at::kLong,
      "topk_indices_buffer must be int32 or int64");
}

at::Tensor FoldQWeights(const at::Tensor& q_quant, const at::Tensor& weights) {
  return (q_quant.to(at::kFloat) * weights.to(at::kFloat).unsqueeze(-1)).sum(1);
}

void FillArangeRow(at::Tensor topk_indices_buffer, int64_t token_idx, int64_t valid_len) {
  if (valid_len <= 0) {
    return;
  }
  at::Tensor values = at::arange(valid_len, topk_indices_buffer.options());
  topk_indices_buffer.select(0, token_idx).slice(0, 0, valid_len).copy_(values);
}

struct TopKCandidate {
  float value = 0.0f;
  int32_t index = 0;
};

void CopyTopKIndicesWithAtenFallback(
    at::Tensor topk_indices_buffer,
    int64_t token_idx,
    at::Tensor row,
    int64_t k_take) {
  if (k_take <= 0) {
    return;
  }
  at::Tensor indices = std::get<1>(at::topk(row, k_take, -1, true, true)).to(topk_indices_buffer.scalar_type());
  topk_indices_buffer.select(0, token_idx).slice(0, 0, k_take).copy_(indices);
}

void SelectTopKSorted(const float* row, int64_t valid_len, int64_t k_take, TopKCandidate* top) {
  for (int64_t i = 0; i < k_take; ++i) {
    top[i] = TopKCandidate{row[i], static_cast<int32_t>(i)};
  }

  int64_t min_pos = 0;
  float min_val = top[0].value;
  for (int64_t i = 1; i < k_take; ++i) {
    if (top[i].value < min_val) {
      min_val = top[i].value;
      min_pos = i;
    }
  }

  for (int64_t i = k_take; i < valid_len; ++i) {
    const float value = row[i];
    if (value <= min_val) {
      continue;
    }
    top[min_pos] = TopKCandidate{value, static_cast<int32_t>(i)};

    min_pos = 0;
    min_val = top[0].value;
    for (int64_t j = 1; j < k_take; ++j) {
      if (top[j].value < min_val) {
        min_val = top[j].value;
        min_pos = j;
      }
    }
  }

  std::sort(top, top + k_take, [](const TopKCandidate& a, const TopKCandidate& b) {
    return a.value > b.value;
  });
}

void CopyTopKIndicesSorted(
    at::Tensor topk_indices_buffer,
    int64_t token_idx,
    const float* row,
    int64_t valid_len,
    int64_t k_take) {
  if (k_take <= 0) {
    return;
  }
  TopKCandidate top[kMaxCustomTopK];
  SelectTopKSorted(row, valid_len, k_take, top);

  if (topk_indices_buffer.scalar_type() == at::kInt) {
    int32_t* out = topk_indices_buffer.data_ptr<int32_t>() + token_idx * topk_indices_buffer.stride(0);
    for (int64_t i = 0; i < k_take; ++i) {
      out[i * topk_indices_buffer.stride(1)] = top[i].index;
    }
    return;
  }

  int64_t* out = topk_indices_buffer.data_ptr<int64_t>() + token_idx * topk_indices_buffer.stride(0);
  for (int64_t i = 0; i < k_take; ++i) {
    out[i * topk_indices_buffer.stride(1)] = top[i].index;
  }
}

}  // namespace

at::Tensor sparse_attn_indexer_prefill_cpp_v0(
    at::Tensor q_quant,
    at::Tensor weights,
    at::Tensor kv_cache,
    at::Tensor topk_indices_buffer,
    int64_t topk_tokens,
    py::object attn_metadata) {
  const bool profile_enabled = ProfileEnabled();
  SparseAttnIndexerProfile profile;
  const TimePoint total_start = Clock::now();
  TimePoint phase_start = Clock::now();
  CheckSparseAttnIndexerInputs(q_quant, weights, kv_cache, topk_indices_buffer, topk_tokens);
  if (profile_enabled) {
    AddElapsed(&profile.input_check_ms, phase_start);
  }

  phase_start = Clock::now();
  const int64_t num_decodes = IntAttr(attn_metadata, "num_decodes");
  const int64_t num_decode_tokens = IntAttr(attn_metadata, "num_decode_tokens");
  TORCH_CHECK(
      num_decodes == 0 && num_decode_tokens == 0,
      "sparse_attn_indexer cpp_v0 currently supports prefill only");

  const int64_t num_tokens = q_quant.size(0);
  const int64_t head_dim = q_quant.size(2);
  const int64_t block_size = kv_cache.size(1);
  if (profile_enabled) {
    AddElapsed(&profile.metadata_ms, phase_start);
  }

  phase_start = Clock::now();
  topk_indices_buffer.slice(0, 0, num_tokens).fill_(-1);
  if (profile_enabled) {
    AddElapsed(&profile.fill_ms, phase_start);
  }

  phase_start = Clock::now();
  const int64_t num_prefills = IntAttr(attn_metadata, "num_prefills");
  if (num_prefills <= 0 || num_tokens == 0) {
    if (profile_enabled) {
      AddElapsed(&profile.metadata_ms, phase_start);
      PrintProfile(profile, ElapsedMs(total_start));
    }
    return topk_indices_buffer;
  }

  py::object prefill_metadata = attn_metadata.attr("prefill");
  py::object chunks = prefill_metadata.attr("chunks");
  if (profile_enabled) {
    AddElapsed(&profile.metadata_ms, phase_start);
  }
  at::Tensor q_w;

  for (py::handle chunk_handle : chunks) {
    py::object chunk = py::reinterpret_borrow<py::object>(chunk_handle);
    const int64_t token_start = IntAttr(chunk, "token_start");
    const int64_t token_end = IntAttr(chunk, "token_end");
    const int64_t num_chunk_tokens = token_end - token_start;
    if (num_chunk_tokens == 0) {
      continue;
    }
    if (profile_enabled) {
      ++profile.chunks;
    }

    phase_start = Clock::now();
    at::Tensor cu_seq_lens_cpu = TensorAttr(chunk, "cu_seq_lens").to(at::kCPU);
    at::Tensor cu_seqlen_ks_cpu = TensorAttr(chunk, "cu_seqlen_ks").to(at::kCPU);
    at::Tensor cu_seqlen_ke_cpu = TensorAttr(chunk, "cu_seqlen_ke").to(at::kCPU);
    at::Tensor valid_lens_cpu = cu_seqlen_ke_cpu - cu_seqlen_ks_cpu;
    const bool use_short_path = valid_lens_cpu.numel() > 0 && valid_lens_cpu.max().item<int64_t>() <= topk_tokens;
    if (profile_enabled) {
      AddElapsed(&profile.metadata_ms, phase_start);
    }

    if (use_short_path) {
      phase_start = Clock::now();
      if (profile_enabled) {
        ++profile.short_chunks;
      }
      for (int64_t i = 0; i < num_chunk_tokens; ++i) {
        const int64_t valid_len = valid_lens_cpu[i].item<int64_t>();
        FillArangeRow(topk_indices_buffer, token_start + i, valid_len);
      }
      if (profile_enabled) {
        AddElapsed(&profile.short_path_ms, phase_start);
      }
      continue;
    }

    phase_start = Clock::now();
    at::Tensor block_table_cpu = TensorAttr(chunk, "block_table").to(at::kCPU);
    const int64_t num_reqs = IntAttr(chunk, "num_reqs");
    const int64_t total_seq_lens = IntAttr(chunk, "total_seq_lens");
    at::Tensor k_gathered = at::empty({total_seq_lens, head_dim}, q_quant.options().dtype(at::kFloat));
    if (profile_enabled) {
      ++profile.scored_chunks;
      AddElapsed(&profile.metadata_ms, phase_start);
    }

    phase_start = Clock::now();
    for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
      const int64_t ks = cu_seq_lens_cpu[req_idx].item<int64_t>();
      const int64_t ke = cu_seq_lens_cpu[req_idx + 1].item<int64_t>();
      const int64_t seq_len = ke - ks;
      if (seq_len == 0) {
        continue;
      }
      const int64_t num_blocks = (seq_len + block_size - 1) / block_size;
      at::Tensor block_ids = block_table_cpu.select(0, req_idx).slice(0, 0, num_blocks).to(at::kLong);
      at::Tensor gathered = kv_cache.index_select(0, block_ids).reshape({num_blocks * block_size, head_dim});
      k_gathered.slice(0, ks, ke).copy_(gathered.slice(0, 0, seq_len).to(at::kFloat));
    }
    if (profile_enabled) {
      AddElapsed(&profile.gather_ms, phase_start);
    }

    if (!q_w.defined()) {
      phase_start = Clock::now();
      q_w = FoldQWeights(q_quant, weights);
      if (profile_enabled) {
        AddElapsed(&profile.fold_q_ms, phase_start);
      }
    }
    phase_start = Clock::now();
    at::Tensor q_w_chunk = q_w.slice(0, token_start, token_end);
    at::Tensor logits = at::matmul(q_w_chunk, k_gathered.t());
    if (profile_enabled) {
      AddElapsed(&profile.matmul_ms, phase_start);
    }

    phase_start = Clock::now();
    TORCH_CHECK(logits.scalar_type() == at::kFloat, "sparse_attn_indexer cpp_v0 logits must be float32");
    TORCH_CHECK(logits.is_contiguous(), "sparse_attn_indexer cpp_v0 logits must be contiguous");
    const float* logits_data = logits.data_ptr<float>();
    const int64_t logits_stride0 = logits.stride(0);
    for (int64_t i = 0; i < num_chunk_tokens; ++i) {
      const int64_t ks_i = cu_seqlen_ks_cpu[i].item<int64_t>();
      const int64_t ke_i = cu_seqlen_ke_cpu[i].item<int64_t>();
      const int64_t valid_len = ke_i - ks_i;
      if (valid_len <= 0) {
        continue;
      }
      const int64_t k_take = std::min<int64_t>(topk_tokens, valid_len);
      if (k_take <= kMaxCustomTopK) {
        const float* row = logits_data + i * logits_stride0 + ks_i;
        CopyTopKIndicesSorted(topk_indices_buffer, token_start + i, row, valid_len, k_take);
      } else {
        at::Tensor row = logits.select(0, i).slice(0, ks_i, ke_i);
        CopyTopKIndicesWithAtenFallback(topk_indices_buffer, token_start + i, row, k_take);
      }
    }
    if (profile_enabled) {
      AddElapsed(&profile.topk_ms, phase_start);
    }
  }

  if (profile_enabled) {
    PrintProfile(profile, ElapsedMs(total_start));
  }
  return topk_indices_buffer;
}
