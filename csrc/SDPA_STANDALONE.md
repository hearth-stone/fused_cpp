# SDPA Standalone Build

Current recommended path for llama.cpp/BGE fp32 is no longer the `csrc`
registry-based standalone build. Use the extracted source under:

```text
standalone/fp32_packqkv/
```

That directory builds a pure C/C++ shared library without modifying the
existing SDPA framework and without linking Python, pybind, or libtorch. Its
required headers are copied under `standalone/fp32_packqkv/csrc`, so it can be
copied and built without the repository-level `csrc` tree.

Build:

```sh
cd standalone/fp32_packqkv
make -j
```

Run the BGE-small fp32 single-core benchmark:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./fp32_packqkv_sdpa_bench \
  --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64 --iters=20 --warmup=5 --noncausal
```

Full llama.cpp integration notes are in:

```text
docs/llama_cpp_sdpa_integration.md
```
