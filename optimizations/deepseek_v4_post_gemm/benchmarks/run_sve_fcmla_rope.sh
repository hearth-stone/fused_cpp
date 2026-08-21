#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/fused_cpp_sve_fcmla_rope}"
CXX="${CXX:-g++}"
SVE_BITS="${FUSED_CPP_MOE_SVE_VECTOR_BITS:-256}"
TARGET="${BUILD_DIR}/bench_sve_fcmla_rope_sve${SVE_BITS}"
SOURCE="${SCRIPT_DIR}/bench_sve_fcmla_rope.cpp"

mkdir -p "${BUILD_DIR}"
if [[ ! -x "${TARGET}" || "${SOURCE}" -nt "${TARGET}" ]]; then
  "${CXX}" -O2 -std=c++17 -fopenmp -Wall -Wextra -Werror \
    -march=armv8.6-a+sve -msve-vector-bits="${SVE_BITS}" \
    "${SOURCE}" -o "${TARGET}"
fi

exec "${TARGET}" "$@"
