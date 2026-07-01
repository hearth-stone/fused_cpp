# Standalone build for the MoE planner core (torch-free).
#
#   make            # build ./moe_planner (single-thread)
#   make OMP=1      # enable OpenMP (needs libomp; threads the heavy planners)
#   make run        # build and run the full sweep
#   make clean
#
# The planner sources live in ../../csrc/moe_planner and are included via
# `#include "moe_planner/..."` relative to the csrc root.

CXX      ?= c++
CSRC     := ../../csrc
STD      := -std=c++17
OPT      := -O3
WARN     := -Wall -Wextra
INC      := -I$(CSRC)

UNAME_M  := $(shell uname -m)
UNAME_S  := $(shell uname -s)

# AArch64 SIMD target flags (mirrors setup.py).
ifeq ($(UNAME_M),arm64)
  ARCH := -mcpu=apple-m2
else ifeq ($(UNAME_M),aarch64)
  ARCH := -march=armv8.6-a+bf16+i8mm$(shell grep -q ' sve' /proc/cpuinfo 2>/dev/null && echo +sve)
else
  ARCH :=
endif

CXXFLAGS := $(STD) $(OPT) $(WARN) $(ARCH) $(INC)
LDFLAGS  :=

# Optional OpenMP. On macOS, prefer Homebrew libomp; on Linux use -fopenmp.
ifeq ($(OMP),1)
  ifeq ($(UNAME_S),Darwin)
    BREW_OMP := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(BREW_OMP),)
      CXXFLAGS += -Xpreprocessor -fopenmp -I$(BREW_OMP)/include
      LDFLAGS  += -L$(BREW_OMP)/lib -lomp -Wl,-rpath,$(BREW_OMP)/lib
    endif
  else
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
  endif
endif

SRCS := main.cpp \
        $(CSRC)/moe_planner/planner_common.cpp \
        $(CSRC)/moe_planner/planner_dispatch.cpp \
        $(CSRC)/moe_planner/plan_fixed_global_threads.cpp \
        $(CSRC)/moe_planner/plan_sorted_token_balanced_1t.cpp \
        $(CSRC)/moe_planner/plan_uniform_waves.cpp \
        $(CSRC)/moe_planner/plan_greedy_marginal_gain.cpp \
        $(CSRC)/moe_planner/plan_enumerate_core_groups.cpp \
        $(CSRC)/moe_planner/plan_load_proportional.cpp \
        $(CSRC)/moe_planner/plan_sqrt_load.cpp \
        $(CSRC)/moe_planner/plan_log_load.cpp \
        $(CSRC)/moe_planner/plan_heavy_light_hybrid.cpp \
        $(CSRC)/moe_planner/plan_karmarkar_karp.cpp

OBJS := $(notdir $(SRCS:.cpp=.o))

VPATH := .:$(CSRC)/moe_planner

moe_planner: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: moe_planner
	./moe_planner

clean:
	rm -f $(OBJS) moe_planner

.PHONY: run clean
