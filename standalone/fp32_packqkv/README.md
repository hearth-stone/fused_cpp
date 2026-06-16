# fp32 packqkv standalone SDPA

This directory extracts the current fp32 `flash2_neon_l3kv_packqkv_pbf16pv`
path into a small standalone C++ target. It does not modify or register into
the existing `csrc` SDPA framework, and it does not link Python, pybind, or
libtorch.

All headers needed by this target are copied into this directory under
`csrc/`; building does not include files from the repository-level `../../csrc`.
You can copy only `standalone/fp32_packqkv` to a clean directory and build it
there.

The implementation keeps the current fp32 compute shape:

- input/output contiguous core layout: `q [B,N,L,E]`, `k [B,N,S,E]`,
  `v [B,N,S,Ev]`, `out [B,N,L,Ev]`
- llama.cpp entry layout: byte-strided `q [D,L,H,B]`, `k [D,S,H,B]`,
  `v [DV,S,H,B]`, and output `[DV,H,L,B]`
- the llama.cpp entry consumes those byte strides directly: Q is read in-place,
  K/V are packed from strided ggml memory, and the kernel writes directly to
  the strided output tensor
- K is fully packed at entry as `[B,N,S/8,E,8]`
- V is packed as `[B,N,Ev/8,S,8]`
- Q remains row-major; QK uses the copied fp32 packed-K 8x8 lane-FMLA trait
  `MK_Fp32PackK8PQuad`
- PV uses the existing fp32 pquad microkernel
- optional llama.cpp F16/F32 mask input uses ggml layout
  `[S,L,H_mask,B_mask]`; the kernel reads it directly and adds it while
  writing QK blocks to the scores tile

Build on Arm:

```sh
cd standalone/fp32_packqkv
make -j
```

Run BGE-small single-core benchmark:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./fp32_packqkv_sdpa_bench \
  --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64 --iters=20 --warmup=5 --noncausal
```

Build only the shared library:

```sh
make lib
```

The shared library exports three llama.cpp-facing C symbols:

```c
int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(...);
int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f16(...);
int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f32(...);
```

The masked entries read F16 or F32 mask values as additive logits:

```text
scores += mask[s, l, h % H_mask, b % B_mask]
```

The masked path does not materialize a fp32 `[B,N,L,S]` mask. Full 8x8 QK
blocks add mask values to the temporary QK block before writing `scores`;
partial blocks add mask immediately after their QK writeback.
Mask blocks that are entirely `0` skip the add. Blocks that are entirely off
(`-inf`, or finite mask values <= `-1000`) skip QK, write `-inf` scores, and
skip the corresponding PV key range when it is fully masked for the current Q
tile.

The source also provides a contiguous pointer helper for the local bench and
for direct source integration, but it is hidden by the shared library version
script. Add it to `fp32_packqkv_sdpa.version` only if a downstream wrapper
really needs to dlsym it.

Useful verification commands:

```sh
make check
nm -D --defined-only libfused_cpp_sdpa_fp32_packqkv.so
ldd libfused_cpp_sdpa_fp32_packqkv.so
```

`nm` should show only the three `fused_cpp_sdpa_*llamacpp*` symbols above. `ldd`
should not show torch, pybind, or Python libraries.

The llama.cpp entry currently assumes `n_head == n_head_kv`, which is the BGE
case. For GQA/MQA, add an explicit `H_kv` parameter and map query head `h` to
KV head `h / (H / H_kv)`.
