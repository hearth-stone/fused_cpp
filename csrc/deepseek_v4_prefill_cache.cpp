// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace {

constexpr int64_t kSparsePrefillTopKAlignment = 128;

void CheckCpuTensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
}

void CheckIntTensor(const at::Tensor& tensor, const char* name) {
  CheckCpuTensor(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
              name, " must be torch.int32 or torch.int64, got ", tensor.scalar_type());
}

int64_t ReadIndex1d(const at::Tensor& tensor, int64_t i) {
  if (tensor.scalar_type() == at::kInt) {
    return static_cast<int64_t>(tensor.data_ptr<int32_t>()[i * tensor.stride(0)]);
  }
  return tensor.data_ptr<int64_t>()[i * tensor.stride(0)];
}

int64_t ReadIndex2d(const at::Tensor& tensor, int64_t i, int64_t j) {
  if (tensor.scalar_type() == at::kInt) {
    return static_cast<int64_t>(
        tensor.data_ptr<int32_t>()[i * tensor.stride(0) + j * tensor.stride(1)]);
  }
  return tensor.data_ptr<int64_t>()[i * tensor.stride(0) + j * tensor.stride(1)];
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct GatherRegion {
  const c10::BFloat16* k_ptr;
  c10::BFloat16* out_ptr;
  const at::Tensor* block_table;
  int64_t batch_idx;
  int64_t start_pos;
  int64_t gather_len;
  int64_t block_size;
  int64_t offset;
  int64_t k_size0;
  int64_t k_s0;
  int64_t k_s1;
  int64_t k_s2;
  int64_t out_s0;
  int64_t out_s1;
  int64_t out_s2;
  int64_t head_dim;
};

struct CombineRegion {
  int64_t batch_idx;
  int64_t query_start;
  int64_t query_len;
  int64_t start_pos;
  int64_t gather_start;
};

inline void CopyBf16Row(const c10::BFloat16* src,
                        int64_t src_stride,
                        c10::BFloat16* dst,
                        int64_t dst_stride,
                        int64_t head_dim) {
  if (head_dim <= 0) {
    return;
  }
  if (src_stride == 1 && dst_stride == 1) {
#if defined(__ARM_FEATURE_SVE)
    const auto* src_u16 = reinterpret_cast<const uint16_t*>(src);
    auto* dst_u16 = reinterpret_cast<uint16_t*>(dst);
    const int64_t vl = static_cast<int64_t>(svcnth());
    const int64_t step = 4 * vl;
    const svbool_t pg_all = svptrue_b16();
    int64_t d = 0;
    for (; d + step <= head_dim; d += step) {
      const svuint16_t v0 = svld1_u16(pg_all, src_u16 + d);
      const svuint16_t v1 = svld1_u16(pg_all, src_u16 + d + vl);
      const svuint16_t v2 = svld1_u16(pg_all, src_u16 + d + 2 * vl);
      const svuint16_t v3 = svld1_u16(pg_all, src_u16 + d + 3 * vl);
      svst1_u16(pg_all, dst_u16 + d, v0);
      svst1_u16(pg_all, dst_u16 + d + vl, v1);
      svst1_u16(pg_all, dst_u16 + d + 2 * vl, v2);
      svst1_u16(pg_all, dst_u16 + d + 3 * vl, v3);
    }
    for (; d < head_dim; d += vl) {
      const svbool_t pg = svwhilelt_b16(d, head_dim);
      const svuint16_t v = svld1_u16(pg, src_u16 + d);
      svst1_u16(pg, dst_u16 + d, v);
    }
#else
    std::memcpy(dst, src, static_cast<size_t>(head_dim) * sizeof(c10::BFloat16));
#endif
    return;
  }

  for (int64_t d = 0; d < head_dim; ++d) {
    dst[d * dst_stride] = src[d * src_stride];
  }
}

#if defined(__ARM_FEATURE_SVE)
inline void CopyAddI32ContiguousSve(const int32_t* src,
                                    int32_t* dst,
                                    int64_t len,
                                    int32_t addend) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svint32_t add_vec = svdup_n_s32(addend);
  int64_t j = 0;
  for (; j < len; j += vl) {
    const svbool_t pg = svwhilelt_b32(j, len);
    const svint32_t values = svld1_s32(pg, src + j);
    svst1_s32(pg, dst + j, svadd_s32_z(pg, values, add_vec));
  }
}

inline void StoreI32ArangeContiguousSve(int32_t* dst,
                                        int64_t len,
                                        int32_t base) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svint32_t lane_offsets = svindex_s32(0, 1);
  int64_t j = 0;
  for (; j < len; j += vl) {
    const svbool_t pg = svwhilelt_b32(j, len);
    const svint32_t base_vec = svdup_n_s32(static_cast<int32_t>(base + j));
    svst1_s32(pg, dst + j, svadd_s32_z(pg, base_vec, lane_offsets));
  }
}
#endif

inline void CopyGatherRow(const GatherRegion& region, int64_t row_idx) {
  const int64_t pos = region.start_pos + row_idx;
  const int64_t block_in_seq = pos / region.block_size;
  const int64_t pos_in_block = pos % region.block_size;
  const int64_t physical_block =
      ReadIndex2d(*region.block_table, region.batch_idx, block_in_seq);
  TORCH_CHECK(physical_block >= 0 && physical_block < region.k_size0,
              "block_table[", region.batch_idx, ", ", block_in_seq,
              "] out of range: ", physical_block,
              " for k_cache.size(0)=", region.k_size0);

  const auto* src =
      region.k_ptr + physical_block * region.k_s0 + pos_in_block * region.k_s1;
  auto* dst = region.out_ptr + region.batch_idx * region.out_s0 +
              (region.offset + row_idx) * region.out_s1;
  CopyBf16Row(src, region.k_s2, dst, region.out_s2, region.head_dim);
}

inline void WriteCombinedIndexRow(const int32_t* topk_row,
                                  int64_t topk_s1,
                                  int32_t* row,
                                  int64_t out_s1,
                                  int64_t topk_len,
                                  int64_t swa_len,
                                  int32_t topk_addend,
                                  int32_t swa_base) {
  if (topk_len > 0) {
    if (topk_s1 == 1 && out_s1 == 1) {
#if defined(__ARM_FEATURE_SVE)
      CopyAddI32ContiguousSve(topk_row, row, topk_len, topk_addend);
#else
      for (int64_t j = 0; j < topk_len; ++j) {
        row[j] = topk_row[j] + topk_addend;
      }
#endif
    } else {
      for (int64_t j = 0; j < topk_len; ++j) {
        row[j * out_s1] = topk_row[j * topk_s1] + topk_addend;
      }
    }
  }

  if (swa_len > 0) {
    int32_t* swa_row = row + topk_len * out_s1;
    if (out_s1 == 1) {
#if defined(__ARM_FEATURE_SVE)
      StoreI32ArangeContiguousSve(swa_row, swa_len, swa_base);
#else
      for (int64_t j = 0; j < swa_len; ++j) {
        swa_row[j] = static_cast<int32_t>(swa_base + j);
      }
#endif
    } else {
      for (int64_t j = 0; j < swa_len; ++j) {
        swa_row[j * out_s1] = static_cast<int32_t>(swa_base + j);
      }
    }
  }
}

void RunGatherRegions(const std::vector<GatherRegion>& regions) {
  if (regions.empty()) {
    return;
  }

  std::vector<int64_t> starts(regions.size() + 1, 0);
  for (size_t i = 0; i < regions.size(); ++i) {
    starts[i + 1] = starts[i] + regions[i].gather_len;
  }
  const int64_t total_rows = starts.back();
  if (total_rows == 0) {
    return;
  }

  at::parallel_for(0, total_rows, 16, [&](int64_t begin, int64_t end) {
    size_t region_idx =
        static_cast<size_t>(std::upper_bound(starts.begin(), starts.end(), begin) -
                            starts.begin() - 1);
    int64_t cursor = begin;
    while (cursor < end) {
      const int64_t region_end = std::min<int64_t>(end, starts[region_idx + 1]);
      const GatherRegion& region = regions[region_idx];
      for (int64_t row = cursor - starts[region_idx];
           row < region_end - starts[region_idx]; ++row) {
        CopyGatherRow(region, row);
      }
      cursor = region_end;
      ++region_idx;
    }
  });
}

void AppendGatherRegions(std::vector<GatherRegion>& regions,
                         const at::Tensor& out,
                         const at::Tensor& k_cache,
                         const at::Tensor& seq_lens,
                         const c10::optional<at::Tensor>& gather_lens,
                         const at::Tensor& block_table,
                         int64_t block_size,
                         int64_t offset,
                         const char* op_name) {
  CheckCpuTensor(out, op_name);
  CheckCpuTensor(k_cache, op_name);
  CheckIntTensor(seq_lens, op_name);
  CheckIntTensor(block_table, op_name);
  TORCH_CHECK(out.dim() == 3, "out must be 3-D [chunk_size, M, head_dim], got ", out.dim(), "-D");
  TORCH_CHECK(k_cache.dim() == 3, "k_cache must be 3-D [num_blocks, block_size, head_dim], got ",
              k_cache.dim(), "-D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D, got ", seq_lens.dim(), "-D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D, got ", block_table.dim(), "-D");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16,
              "out must be torch.bfloat16, got ", out.scalar_type());
  TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16,
              "k_cache must be torch.bfloat16, got ", k_cache.scalar_type());
  TORCH_CHECK(block_size > 0, "block_size must be positive, got ", block_size);
  TORCH_CHECK(offset >= 0, "offset must be non-negative, got ", offset);
  TORCH_CHECK(k_cache.size(1) == block_size,
              "block_size mismatch: argument block_size=", block_size,
              " k_cache.size(1)=", k_cache.size(1));
  TORCH_CHECK(out.size(2) == k_cache.size(2),
              "head_dim mismatch: out=", out.size(2), " k_cache=", k_cache.size(2));

  const int64_t chunk_size = seq_lens.size(0);
  TORCH_CHECK(out.size(0) >= chunk_size,
              "out first dimension must cover seq_lens: out.size(0)=", out.size(0),
              " seq_lens=", chunk_size);
  TORCH_CHECK(block_table.size(0) >= chunk_size,
              "block_table first dimension must cover seq_lens: block_table.size(0)=",
              block_table.size(0), " seq_lens=", chunk_size);

  at::Tensor gather_lens_tensor;
  if (gather_lens.has_value()) {
    gather_lens_tensor = gather_lens.value();
    CheckIntTensor(gather_lens_tensor, op_name);
    TORCH_CHECK(gather_lens_tensor.dim() == 1,
                "gather_lens must be 1-D, got ", gather_lens_tensor.dim(), "-D");
    TORCH_CHECK(gather_lens_tensor.size(0) >= chunk_size,
                "gather_lens must cover seq_lens: gather_lens.size(0)=",
                gather_lens_tensor.size(0), " seq_lens=", chunk_size);
  }

  if (k_cache.numel() == 0 || chunk_size == 0) {
    return;
  }

  const int64_t head_dim = k_cache.size(2);
  const auto* k_ptr = k_cache.data_ptr<c10::BFloat16>();
  auto* out_ptr = out.data_ptr<c10::BFloat16>();
  const int64_t out_s0 = out.stride(0);
  const int64_t out_s1 = out.stride(1);
  const int64_t out_s2 = out.stride(2);
  const int64_t k_s0 = k_cache.stride(0);
  const int64_t k_s1 = k_cache.stride(1);
  const int64_t k_s2 = k_cache.stride(2);

  for (int64_t batch_idx = 0; batch_idx < chunk_size; ++batch_idx) {
    const int64_t seq_len = ReadIndex1d(seq_lens, batch_idx);
    TORCH_CHECK(seq_len >= 0, "seq_lens[", batch_idx, "] must be non-negative, got ", seq_len);
    if (seq_len == 0) {
      continue;
    }

    const int64_t gather_len =
        gather_lens.has_value() ? ReadIndex1d(gather_lens_tensor, batch_idx) : seq_len;
    TORCH_CHECK(gather_len >= 0,
                "gather_lens[", batch_idx, "] must be non-negative, got ", gather_len);
    TORCH_CHECK(gather_len <= seq_len,
                "gather_lens[", batch_idx, "]=", gather_len,
                " exceeds seq_lens[", batch_idx, "]=", seq_len);
    if (gather_len == 0) {
      continue;
    }
    TORCH_CHECK(offset + gather_len <= out.size(1),
                "out M dimension is too small: offset=", offset,
                " gather_len=", gather_len, " out.size(1)=", out.size(1));

    const int64_t needed_blocks = (seq_len + block_size - 1) / block_size;
    TORCH_CHECK(needed_blocks <= block_table.size(1),
                "block_table row is too short: need ", needed_blocks,
                " blocks, has ", block_table.size(1));

    const int64_t start_pos = seq_len - gather_len;
    regions.push_back(GatherRegion{
        k_ptr,
        out_ptr,
        &block_table,
        batch_idx,
        start_pos,
        gather_len,
        block_size,
        offset,
        k_cache.size(0),
        k_s0,
        k_s1,
        k_s2,
        out_s0,
        out_s1,
        out_s2,
        head_dim,
    });
  }
}

void GatherKCacheImpl(const at::Tensor& out,
                      const at::Tensor& k_cache,
                      const at::Tensor& seq_lens,
                      const c10::optional<at::Tensor>& gather_lens,
                      const at::Tensor& block_table,
                      int64_t block_size,
                      int64_t offset,
                      const char* op_name) {
  std::vector<GatherRegion> regions;
  AppendGatherRegions(regions, out, k_cache, seq_lens, gather_lens, block_table,
                      block_size, offset, op_name);
  RunGatherRegions(regions);
}

}  // namespace

void deepseek_v4_dequantize_and_gather_k_cache(at::Tensor out,
                                               at::Tensor k_cache,
                                               at::Tensor seq_lens,
                                               c10::optional<at::Tensor> gather_lens,
                                               at::Tensor block_table,
                                               int64_t block_size,
                                               int64_t offset) {
  GatherKCacheImpl(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset,
                   "deepseek_v4_dequantize_and_gather_k_cache");
}

void deepseek_v4_dequantize_and_gather_dual_k_cache(at::Tensor out,
                                                    at::Tensor compressed_k_cache,
                                                    at::Tensor compressed_seq_lens,
                                                    at::Tensor compressed_block_table,
                                                    int64_t compressed_block_size,
                                                    int64_t compressed_offset,
                                                    bool has_compressed,
                                                    at::Tensor swa_k_cache,
                                                    at::Tensor swa_seq_lens,
                                                    at::Tensor swa_gather_lens,
                                                    at::Tensor swa_block_table,
                                                    int64_t swa_block_size,
                                                    int64_t swa_offset) {
  std::vector<GatherRegion> regions;
  if (has_compressed) {
    AppendGatherRegions(regions, out, compressed_k_cache, compressed_seq_lens, c10::nullopt,
                        compressed_block_table, compressed_block_size, compressed_offset,
                        "deepseek_v4_dequantize_and_gather_dual_k_cache compressed");
  }
  const c10::optional<at::Tensor> swa_gather_lens_opt(swa_gather_lens);
  AppendGatherRegions(regions, out, swa_k_cache, swa_seq_lens, swa_gather_lens_opt,
                      swa_block_table, swa_block_size, swa_offset,
                      "deepseek_v4_dequantize_and_gather_dual_k_cache swa");
  RunGatherRegions(regions);
}

std::tuple<at::Tensor, at::Tensor> deepseek_v4_combine_topk_swa_indices(
    at::Tensor topk_indices,
    at::Tensor query_start_loc,
    at::Tensor seq_lens,
    at::Tensor gather_lens,
    int64_t window_size,
    int64_t compress_ratio,
    int64_t topk,
    int64_t M,
    int64_t N) {
  CheckCpuTensor(topk_indices, "deepseek_v4_combine_topk_swa_indices topk_indices");
  CheckIntTensor(query_start_loc, "deepseek_v4_combine_topk_swa_indices query_start_loc");
  CheckIntTensor(seq_lens, "deepseek_v4_combine_topk_swa_indices seq_lens");
  CheckIntTensor(gather_lens, "deepseek_v4_combine_topk_swa_indices gather_lens");
  TORCH_CHECK(topk_indices.dim() == 2,
              "topk_indices must be 2-D [num_tokens, K], got ", topk_indices.dim(), "-D");
  TORCH_CHECK(topk_indices.scalar_type() == at::kInt,
              "topk_indices must be torch.int32, got ", topk_indices.scalar_type());
  TORCH_CHECK(query_start_loc.dim() == 1,
              "query_start_loc must be 1-D, got ", query_start_loc.dim(), "-D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D, got ", seq_lens.dim(), "-D");
  TORCH_CHECK(gather_lens.dim() == 1,
              "gather_lens must be 1-D, got ", gather_lens.dim(), "-D");
  TORCH_CHECK(window_size >= 0, "window_size must be non-negative, got ", window_size);
  TORCH_CHECK(topk >= 0, "topk must be non-negative, got ", topk);
  TORCH_CHECK(compress_ratio >= 0, "compress_ratio must be non-negative, got ", compress_ratio);
  TORCH_CHECK(M >= 0, "M must be non-negative, got ", M);
  TORCH_CHECK(N >= 0, "N must be non-negative, got ", N);

  const int64_t num_tokens = topk_indices.size(0);
  const int64_t topk_width = topk_indices.size(1);
  const int64_t num_reqs = seq_lens.size(0);
  TORCH_CHECK(gather_lens.size(0) >= num_reqs,
              "gather_lens must cover seq_lens: gather_lens.size(0)=",
              gather_lens.size(0), " seq_lens=", num_reqs);
  TORCH_CHECK(query_start_loc.size(0) >= num_reqs + 1,
              "query_start_loc must have num_reqs + 1 entries: query_start_loc.size(0)=",
              query_start_loc.size(0), " num_reqs=", num_reqs);
  TORCH_CHECK(topk <= topk_width,
              "topk=", topk, " exceeds topk_indices width=", topk_width);

  const int64_t combined_topk =
      AlignUp(topk + window_size, kSparsePrefillTopKAlignment);
  at::Tensor combined_indices = at::full(
      {num_tokens, combined_topk},
      -1,
      topk_indices.options().dtype(at::kInt));
  at::Tensor combined_lens = at::zeros(
      {num_tokens},
      topk_indices.options().dtype(at::kInt));
  if (num_tokens == 0) {
    return std::make_tuple(combined_indices, combined_lens);
  }

  const int64_t base = ReadIndex1d(query_start_loc, 0);
  const auto* topk_ptr = topk_indices.data_ptr<int32_t>();
  auto* out_ptr = combined_indices.data_ptr<int32_t>();
  auto* lens_ptr = combined_lens.data_ptr<int32_t>();
  const int64_t topk_s0 = topk_indices.stride(0);
  const int64_t topk_s1 = topk_indices.stride(1);
  const int64_t out_s0 = combined_indices.stride(0);
  const int64_t out_s1 = combined_indices.stride(1);
  const int64_t lens_s0 = combined_lens.stride(0);

  std::vector<CombineRegion> regions;
  regions.reserve(static_cast<size_t>(num_reqs));
  std::vector<int64_t> starts;
  starts.reserve(static_cast<size_t>(num_reqs) + 1);
  starts.push_back(0);

  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t query_start = ReadIndex1d(query_start_loc, batch_idx) - base;
    const int64_t query_end = ReadIndex1d(query_start_loc, batch_idx + 1) - base;
    TORCH_CHECK(query_start >= 0 && query_start <= query_end && query_end <= num_tokens,
                "invalid chunk-local query range for batch ", batch_idx,
                ": [", query_start, ", ", query_end, ") num_tokens=", num_tokens);
    const int64_t query_len = query_end - query_start;
    if (query_len == 0) {
      continue;
    }

    const int64_t seq_len = ReadIndex1d(seq_lens, batch_idx);
    const int64_t gather_len = ReadIndex1d(gather_lens, batch_idx);
    TORCH_CHECK(seq_len >= 0, "seq_lens[", batch_idx, "] must be non-negative, got ", seq_len);
    TORCH_CHECK(gather_len >= 0,
                "gather_lens[", batch_idx, "] must be non-negative, got ", gather_len);
    TORCH_CHECK(query_len <= seq_len,
                "query_len=", query_len, " exceeds seq_lens[", batch_idx, "]=", seq_len);
    TORCH_CHECK(gather_len <= seq_len,
                "gather_lens[", batch_idx, "]=", gather_len,
                " exceeds seq_lens[", batch_idx, "]=", seq_len);

    const int64_t start_pos = seq_len - query_len;
    const int64_t gather_start = seq_len - gather_len;
    regions.push_back(CombineRegion{
        batch_idx,
        query_start,
        query_len,
        start_pos,
        gather_start,
    });
    starts.push_back(starts.back() + query_len);
  }

  const int64_t total_query_tokens = starts.back();
  if (total_query_tokens == 0) {
    return std::make_tuple(combined_indices, combined_lens);
  }

  at::parallel_for(0, total_query_tokens, 64, [&](int64_t begin, int64_t end) {
    size_t region_idx =
        static_cast<size_t>(std::upper_bound(starts.begin(), starts.end(), begin) -
                            starts.begin() - 1);
    int64_t cursor = begin;
    while (cursor < end) {
      const int64_t region_end = std::min<int64_t>(end, starts[region_idx + 1]);
      const CombineRegion& region = regions[region_idx];
      const int32_t topk_addend = static_cast<int32_t>(M * region.batch_idx);
      for (int64_t local = cursor - starts[region_idx];
           local < region_end - starts[region_idx]; ++local) {
        const int64_t token_idx = region.query_start + local;
        const int64_t pos = region.start_pos + local;
        const int64_t topk_len =
            compress_ratio > 0 ? std::min((pos + 1) / compress_ratio, topk) : 0;
        const int64_t swa_len = std::min(pos + 1, window_size);
        const int64_t combined_len = topk_len + swa_len;
        if (combined_len > 0) {
          auto* row = out_ptr + token_idx * out_s0;
          const auto* topk_row =
              topk_len > 0 ? topk_ptr + token_idx * topk_s0 : topk_ptr;
          const int32_t swa_base =
              static_cast<int32_t>(M * region.batch_idx + N + pos - swa_len + 1 -
                                   region.gather_start);

          WriteCombinedIndexRow(topk_row, topk_s1, row, out_s1, topk_len, swa_len,
                                topk_addend, swa_base);
        }
        lens_ptr[token_idx * lens_s0] = static_cast<int32_t>(combined_len);
      }
      cursor = region_end;
      ++region_idx;
    }
  });

  return std::make_tuple(combined_indices, combined_lens);
}
