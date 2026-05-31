#!/usr/bin/env bash
# build.sh — compile microkernel_min on macOS Apple Silicon or Linux aarch64.
set -e
cd "$(dirname "$0")"

UNAME=$(uname -s)
ARCH=$(uname -m)

if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "aarch64" ]; then
  echo "ERROR: This requires aarch64 (got $ARCH)"
  exit 1
fi

if [ "$UNAME" = "Darwin" ]; then
  CXX=${CXX:-clang++}
  # Apple Silicon (M1+) 有 BFMMLA，但 Apple clang 不会从 -mcpu=apple-m1 自动
  # 启用 bf16 target feature；显式加 +bf16+i8mm 触发 vbfmmlaq_f32 等 intrinsic。
  EXTRA="-mcpu=apple-m1+bf16+i8mm"
elif [ "$UNAME" = "Linux" ]; then
  CXX=${CXX:-g++}
  # Neoverse / 鲲鹏 都支持 v8.6+bf16 子集
  EXTRA="-march=armv8.6-a+bf16"
else
  echo "ERROR: Unsupported OS $UNAME"
  exit 1
fi

set -x
$CXX -O3 -std=c++17 $EXTRA \
     -Wall -Wno-unused-function \
     main.cpp -o microkernel_min
set +x

echo "Built: ./microkernel_min  (CXX=$CXX, flags=$EXTRA)"
