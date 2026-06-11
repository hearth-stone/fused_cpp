# Clean standalone SDPA

This directory is a stripped standalone C++ version of the current best bf16
SDPA path:

- QK: `gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner`
- PV: `gemm_pv_microkernel_8x8_bf16_pbf16_prepacked`
- softmax writes bf16 `P` scratch for full 8-row blocks
- causal softmax only processes the valid prefix and zeroes the `P` tail

The compute path is intentionally not a second copy: the standalone wrapper
packs `K/V`, constructs `SdpaParams` and `TileSizes`, then calls the same
`run_path_collapse3_packqkv<..., kPbf16PV=true>` / `process_q_tile_lc_packqkv`
templates used by the extension.

What is intentionally removed:

- PyTorch/ATen tensor entry points
- SDPA version registry and string dispatch
- fp32 path
- mask path
- generic SDPA version dispatch
- path-B fallback and adaptive multi-kernel plumbing

The public input layout is contiguous bf16:

- `Q[B, N, L, E]`
- `K[B, N, S, E]`
- `V[B, N, S, Ev]`
- output is fp32 `O[B, N, L, Ev]`

Current constraints:

- `S % 8 == 0`
- `Ev % 8 == 0`
- no additive mask

Build on an AArch64 BF16 target:

```bash
make
```

The default Makefile enables OpenMP and uses flags close to the PyTorch
extension build. For a compiler without OpenMP, use:

```bash
make USE_OPENMP=0
```

Run a BGE-small-like shape:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close taskset -c 80 \
  ./clean_sdpa_bench --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64 --noncausal
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close taskset -c 80 \
  ./clean_sdpa_bench --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64 --causal
```

Run the small correctness check:

```bash
make check
```

If the compiler needs different architecture flags, override `ARCH_FLAGS`:

```bash
make ARCH_FLAGS="-mcpu=native"
```
