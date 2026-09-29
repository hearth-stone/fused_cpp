# fused_cpp: DeepSeek V4 CPU kernels for vLLM 0.28.0

This branch packages the fused_cpp operators used by the DeepSeek V4 CPU path in
vLLM 0.28.0. It includes the Flash BF16 and W8A8 INT8 paths, including the
W8A8 kernels needed by the Pro model. The supported build target is Linux
AArch64 with BF16, I8MM, and SVE.

## Build

Use the same Python environment and PyTorch installation as vLLM:

```bash
git submodule update --init 3rdparty/xbyak_aarch64
MAX_JOBS=16 /path/to/vllm/.venv/bin/python setup.py build_ext --inplace
```

The zlgemm sources required for the native kernels are included in
`third_party/zlgemm`. The build detects the host's SVE vector length; set
`FUSED_CPP_SVE_VECTOR_BITS` only when building for a different supported vector
length. Put this checkout's `src` directory before other fused_cpp installations
on `PYTHONPATH` when launching vLLM.

## MoE calibration

The vLLM DeepSeek V4 CPU adapter calibrates each tensor-parallel rank during
initialization when no rank-local profile is supplied. It creates a temporary
rank-specific directory under the system temporary directory. A launch script
does not need to generate calibration data first. The explicit profile setting
remains available for controlled restarts or reproducible comparisons.

The fused_cpp API also exposes `enable_moe_planner_quick` and
`calibrate_moe_planner_quick` for applications that initialize the planner
without vLLM.

## Validation

```bash
uv pip install --python /path/to/vllm/.venv/bin/python pytest
PYTHONPATH=src:. /path/to/vllm/.venv/bin/python -m pytest -q \
  -m 'not bench and not slow' tests
```

Model-level performance comparisons should use the same vLLM source, NUMA
binding, model checkpoint, prompt token IDs, request spacing, and warmup for
both builds. This matters because an already warm prefix cache and different
process state can change 2k-prompt latency by more than a kernel edit.
