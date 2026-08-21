#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/fused_cpp_woa_prepacked_gemm}"
CXX="${CXX:-g++}"
SVE_BITS="${FUSED_CPP_MOE_SVE_VECTOR_BITS:-128}"
TARGET="${BUILD_DIR}/bench_prepacked_gemm_sve${SVE_BITS}"

mkdir -p "${BUILD_DIR}"

SOURCES=(
  "${SCRIPT_DIR}/bench_prepacked_gemm.cpp"
  "${ROOT}/csrc/moe/arm/sve_bf16/jit_kernels.cpp"
  "${ROOT}/3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp"
  "${ROOT}/3rdparty/xbyak_aarch64/src/util_impl.cpp"
)
DEPENDENCIES=(
  "${SOURCES[@]}"
  "${ROOT}/csrc/moe/arm/common/nm_window_schedule.h"
  "${ROOT}/csrc/moe/arm/sve_bf16/jit_kernels.h"
  "${ROOT}/csrc/moe/arm/sve_bf16/vector_length.h"
)

REBUILD=0
if [[ ! -x "${TARGET}" ]]; then
  REBUILD=1
else
  for DEPENDENCY in "${DEPENDENCIES[@]}"; do
    if [[ "${DEPENDENCY}" -nt "${TARGET}" ]]; then
      REBUILD=1
      break
    fi
  done
fi

if [[ "${REBUILD}" == 1 ]]; then
  "${CXX}" \
    -O2 \
    -std=c++17 \
    -pthread \
    -fopenmp \
    -march=armv8.6-a+sve+bf16+i8mm \
    "-msve-vector-bits=${SVE_BITS}" \
    -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 \
    "-DFUSED_CPP_MOE_SVE_VECTOR_BITS=${SVE_BITS}" \
    -I"${ROOT}/csrc" \
    -I"${ROOT}/refs/i8gemm/lib" \
    -I"${ROOT}/3rdparty/xbyak_aarch64" \
    -I"${ROOT}/3rdparty/xbyak_aarch64/src" \
    -I"${ROOT}/3rdparty/xbyak_aarch64/xbyak_aarch64" \
    "${SOURCES[@]}" \
    -o "${TARGET}"
fi

exec "${TARGET}" "$@"
