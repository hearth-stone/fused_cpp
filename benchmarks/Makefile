# ── Standalone build for fused_cpp/benchmarks/bench_microkernel_l1 ──────────
#
# 与 setup.py 的 CppExtension 共享同一份编译标志（-mcpu / -march / -O3 /
# -fopenmp），但**不**走 setup.py — 这是一个独立 binary，便于 perf record /
# objdump / lldb 微观分析，启动开销最小。
#
# 链接 libtorch 中的 c10 + torch_cpu，原因：micro-kernel 头声明用的是
# at::BFloat16，绕开成本太高（要复制类型 shim）。Linux/macOS 的 libtorch
# 路径都从当前 Python 解释器的 ``import torch`` 自动抽取。
#
# 用法：
#   make                              # 默认构建 release
#   make CXX=clang++                  # 指定编译器
#   make FUSED_CPP_TARGET_CPU=neoverse-n1
#   make clean

PY ?= python3

# ── 编译器 ──
ifeq ($(shell uname -s),Darwin)
  CXX ?= clang++
else
  CXX ?= g++
endif

# ── libtorch include / lib 路径（从 Python 抽取）──
TORCH_INCLUDE_PATHS := $(shell $(PY) -c "from torch.utils.cpp_extension import include_paths; print('\n'.join(include_paths()))")
TORCH_LIB_PATH := $(shell $(PY) -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
TORCH_INCLUDES := $(addprefix -isystem ,$(TORCH_INCLUDE_PATHS))

# ── 平台相关编译/链接标志（与 setup.py 340-349 行一致）──
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifneq ($(FUSED_CPP_TARGET_CPU),)
  ifneq ($(filter apple-% cortex-% neoverse-%,$(FUSED_CPP_TARGET_CPU)),)
    ARCH_FLAGS := -mcpu=$(FUSED_CPP_TARGET_CPU)
  else
    ARCH_FLAGS := -march=$(FUSED_CPP_TARGET_CPU)
  endif
else ifeq ($(UNAME_S),Darwin)
  ARCH_FLAGS := -mcpu=apple-m2
else
  ifeq ($(UNAME_M),aarch64)
    ARCH_FLAGS := -march=armv8.6-a+bf16+i8mm
  else
    ARCH_FLAGS :=
  endif
endif

# OpenMP 不需要：micro-bench 是单线程逐 op 计时，不调度 OMP。
# torch headers 中的 OMP pragma 只在编译时定义 _OPENMP 才生效，缺省即可。

CXXFLAGS := -std=c++17 -O3 $(ARCH_FLAGS) -DNDEBUG \
            -Wall -Wno-unused-function -Wno-unused-variable \
            -I../csrc $(TORCH_INCLUDES)
LDFLAGS := -L$(TORCH_LIB_PATH) -lc10 -ltorch_cpu \
           -Wl,-rpath,$(TORCH_LIB_PATH)

TARGET := bench_microkernel_l1
SOURCES := bench_microkernel_l1.cpp

.PHONY: all clean print-config

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "  CXX  $@  (CXX=$(CXX))"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

print-config:
	@echo "CXX             = $(CXX)"
	@echo "ARCH_FLAGS      = $(ARCH_FLAGS)"
	@echo "TORCH_LIB_PATH  = $(TORCH_LIB_PATH)"
	@echo "TORCH_INCLUDES  = $(TORCH_INCLUDES)"
	@echo "CXXFLAGS        = $(CXXFLAGS)"
	@echo "LDFLAGS         = $(LDFLAGS)"
