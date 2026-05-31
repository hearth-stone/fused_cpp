python -c "
from fused_cpp import _C
print('=== fused_cpp build info ===')
print('has_openmp:', _C.has_openmp())
print()
print('=== runtime omp ===')
info = _C.get_omp_runtime_info()
for k, v in info.items():
    print(f'  {k}: {v}')
print()
import torch
print('=== torch info ===')
print('torch threads:', torch.get_num_threads())
print('torch parallel info:')
print(torch.__config__.parallel_info())
"
  
for N in 1 4 16 64 79 160; do
    OMP_NUM_THREADS=$N python -c "
import torch, time
from fused_cpp import _C
from fused_cpp.sdpa import sdpa_versioned

print(f'  omp_max in runtime: {_C.get_omp_runtime_info()[\"max_threads\"]}')

q = torch.randn(1, 32, 2048, 192, dtype=torch.bfloat16)
k = torch.randn(1, 32, 2048, 192, dtype=torch.bfloat16)
v = torch.randn(1, 32, 2048, 128, dtype=torch.bfloat16)

for _ in range(3):
    sdpa_versioned(q, k, v, version='flash2_neon_l3kv_packqkv', is_causal=True)

t0 = time.perf_counter()
for _ in range(5):
    sdpa_versioned(q, k, v, version='flash2_neon_l3kv_packqkv', is_causal=True)
dt = (time.perf_counter() - t0) / 5 * 1000
print(f'OMP=$N: packqkv causal = {dt:.1f} ms')
" 2>&1 | grep -E "OMP=|omp_max"
done



SDPA_VERSIONS="pytorch_sdpa,flash2_neon_l3kv,flash2_neon_l3kv_packv,flash2_neon_l3kv_packqkv" \
    DTYPES=bf16 CAUSAL=causal \
    bash bench/sdpa/run_dsr1_tp4_l3_sweep.sh