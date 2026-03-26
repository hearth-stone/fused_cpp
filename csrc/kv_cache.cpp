#include <torch/extension.h>
#include "utils.h"

void write_kv_cache(at::Tensor kv_c, at::Tensor k_pe,
                    at::Tensor kv_cache, at::Tensor slot_mapping) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(kv_c.dim() == 2,
                "write_kv_cache: kv_c must be 2-D [num_tokens, kv_lora_rank]");
    TORCH_CHECK(k_pe.dim() == 2,
                "write_kv_cache: k_pe must be 2-D [num_tokens, qk_rope_head_dim]");
    TORCH_CHECK(kv_cache.dim() == 3,
                "write_kv_cache: kv_cache must be 3-D [num_blocks, block_size, head_size]");
    TORCH_CHECK(slot_mapping.dim() == 1,
                "write_kv_cache: slot_mapping must be 1-D [num_tokens]");
    TORCH_CHECK(kv_c.size(0) == k_pe.size(0),
                "write_kv_cache: kv_c and k_pe num_tokens mismatch (",
                kv_c.size(0), " vs ", k_pe.size(0), ")");
    TORCH_CHECK(kv_c.size(0) == slot_mapping.size(0),
                "write_kv_cache: kv_c and slot_mapping num_tokens mismatch (",
                kv_c.size(0), " vs ", slot_mapping.size(0), ")");
    TORCH_CHECK(kv_c.size(1) + k_pe.size(1) == kv_cache.size(2),
                "write_kv_cache: kv_lora_rank + qk_rope_head_dim (",
                kv_c.size(1) + k_pe.size(1),
                ") must match kv_cache head_size (", kv_cache.size(2), ")");

    auto kv_combined = at::cat({kv_c, k_pe}, /*dim=*/-1);
    auto block_size = kv_cache.size(1);
    auto num_tokens = slot_mapping.size(0);

    auto slot_mapping_i64 = ensure_i64(slot_mapping);
    auto slot_acc = slot_mapping_i64.accessor<int64_t, 1>();
    for (int64_t i = 0; i < num_tokens; ++i) {
        auto slot = slot_acc[i];
        if (slot < 0) {
            continue;
        }
        kv_cache[slot / block_size][slot % block_size] = kv_combined[i];
    }
}

at::Tensor gather_kv_cache(at::Tensor kv_cache, at::Tensor block_table,
                           int64_t seq_len, int64_t block_size) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(kv_cache.dim() == 3,
                "gather_kv_cache: kv_cache must be 3-D [num_blocks, block_size, head_size]");
    TORCH_CHECK(block_table.dim() == 1,
                "gather_kv_cache: block_table must be 1-D [max_blocks]");
    TORCH_CHECK(seq_len > 0,
                "gather_kv_cache: seq_len must be positive");
    TORCH_CHECK(block_size > 0,
                "gather_kv_cache: block_size must be positive");

    auto head_size = kv_cache.size(2);
    auto gathered = at::empty({seq_len, head_size}, kv_cache.options());

    auto num_full_blocks = seq_len / block_size;
    auto remainder = seq_len % block_size;

    auto block_table_i64 = ensure_i64(block_table);
    auto bt_acc = block_table_i64.accessor<int64_t, 1>();
    for (int64_t block_idx = 0; block_idx < num_full_blocks; ++block_idx) {
        auto block_num = bt_acc[block_idx];
        auto start = block_idx * block_size;
        gathered.slice(0, start, start + block_size) = kv_cache[block_num];
    }

    if (remainder > 0) {
        auto block_num = bt_acc[num_full_blocks];
        auto start = num_full_blocks * block_size;
        gathered.slice(0, start, start + remainder) =
            kv_cache[block_num].slice(0, 0, remainder);
    }

    return gathered;
}
