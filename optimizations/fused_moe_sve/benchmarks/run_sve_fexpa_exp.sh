#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/fused_cpp_sve_fexpa_exp}"
CXX="${CXX:-g++}"
SVE_BITS="${FUSED_CPP_MOE_SVE_VECTOR_BITS:-128}"
TARGET="${BUILD_DIR}/bench_sve_fexpa_exp_sve${SVE_BITS}"

mkdir -p "${BUILD_DIR}"

SOURCES=(
  "${SCRIPT_DIR}/bench_sve_fexpa_exp.cpp"
  "${SCRIPT_DIR}/sve_fexpa_exp.S"
)

REBUILD=0
if [[ ! -x "${TARGET}" ]]; then
  REBUILD=1
else
  for SOURCE in "${SOURCES[@]}"; do
    if [[ "${SOURCE}" -nt "${TARGET}" ]]; then
      REBUILD=1
      break
    fi
  done
fi

if [[ "${REBUILD}" == 1 ]]; then
  "${CXX}" -O2 -std=c++17 -fopenmp -Wall -Wextra -Werror \
    -march=armv8.6-a+sve -msve-vector-bits="${SVE_BITS}" \
    -DFUSED_CPP_MOE_SVE_VECTOR_BITS="${SVE_BITS}" \
    "${SOURCES[@]}" -o "${TARGET}"
fi

exec "${TARGET}" "$@"
