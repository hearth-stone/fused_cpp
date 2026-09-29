# Public contracts for the DeepSeek V4 CPU delivery

The explicit `__all__` lists in `src/fused_cpp/__init__.py` and
`src/fused_cpp/moe/__init__.py` define the Python import surface in this
branch. The vLLM 0.28.0 DeepSeek V4 CPU integration uses the operator modules
for BF16 linear layers, W8A8 linear layers, attention GEMM, mHC, sparse MLA,
cache updates, and fused MoE.

For exported callables, preserve accepted tensor shape, dtype, device, layout,
keyword defaults, output shape and dtype, and documented error behavior. Packed
BF16, W8A16, and W8A8 expert weights are opaque objects: a prepared object is
valid for matching operators in the same build and CPU vector length. Its
in-memory layout is an internal kernel contract rather than a serialization
format.

`fused_cpp._C` and `fused_cpp._moe_C` are implementation modules. Test-only
native bindings are diagnostic surfaces. Native assembly, JIT, and translation
unit interfaces must change together with their callers. This delivery does
not expose a standalone C ABI.

The Arm MoE backends used by this branch are `arm_neon_bf16`, `arm_sve_bf16`,
and the SVE/I8MM W8A8 path. Runtime SVE vector length must match the extension
build. Flash BF16 remains the default packed MoE path; W8A8 requires its
matching prepared weight type and Plan V2 execution contract.

`MoePlannerRuntime`, `calibrate_moe_planner_quick`, and
`enable_moe_planner_quick` are the supported planner initialization APIs.
Calibration is synchronous and shape/CPU bound. The vLLM DeepSeek V4 CPU
adapter performs rank-local calibration during worker initialization when no
explicit profile is provided, storing its temporary profile under the system
temporary directory. Importing fused_cpp by itself does not calibrate.

The bitwise and performance acceptance tests for this branch compare the
Flash BF16 and W8A8 INT8 paths against the 0.28.0 baseline on the same Arm
host. Changes to backend numerics require explicit validation in the relevant
operator and model tests.
