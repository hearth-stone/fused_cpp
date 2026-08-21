#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/fused_cpp_w8a8_i8mm}"
SVE_BITS="${FUSED_CPP_MOE_SVE_VECTOR_BITS:-256}"
SOURCE="${ROOT_DIR}/optimizations/fused_moe_sve/benchmarks/bench_w8a8_gemm.cpp"
TARGET="${BUILD_DIR}/bench_w8a8_gemm_sve${SVE_BITS}"
I8_DIR="${ROOT_DIR}/refs/i8gemm/lib"
I8_SOURCES=(
  i8gemm_sve.c
  i8gemm_sve.S
  i8gemm_hybrid.S
  i8gemm_msplit.c
  i8gemm_msplit_k.S
  i8gemm_pack_a_neon.S
)
I8_OBJECTS=()

mkdir -p "${BUILD_DIR}"
relink=0
for source_name in "${I8_SOURCES[@]}"; do
  source_path="${I8_DIR}/${source_name}"
  object_path="${BUILD_DIR}/${source_name//\//_}.o"
  I8_OBJECTS+=("${object_path}")
  if [[ ! -f "${object_path}" || "${source_path}" -nt "${object_path}" ]]; then
    cc -O2 -fPIC -fopenmp -march=armv8.6-a+sve+i8mm -msve-vector-bits="${SVE_BITS}" \
      -I"${I8_DIR}" -c "${source_path}" -o "${object_path}"
    relink=1
  fi
done
if [[ ! -x "${TARGET}" || "${SOURCE}" -nt "${TARGET}" || "${relink}" -eq 1 ]]; then
  c++ -O2 -std=c++17 -fopenmp -Wall -Wextra -Werror \
    -march=armv8.6-a+sve+bf16+i8mm -msve-vector-bits="${SVE_BITS}" \
    -I"${I8_DIR}" "${SOURCE}" "${I8_OBJECTS[@]}" -o "${TARGET}"
fi

exec "${TARGET}" "$@"
