CXX ?= c++
ARCH_FLAGS ?= -march=armv8.6-a+sve+bf16+i8mm
CXXFLAGS ?= -pthread -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -Wno-class-memaccess -O3 -fPIC -std=c++17
CPPFLAGS ?= -I. -I../../csrc/sdpa_microkernels
LDFLAGS ?=
USE_OPENMP ?= 1

ifeq ($(USE_OPENMP),1)
CXXFLAGS += -DFUSED_CPP_HAS_OMP=1 -fopenmp
LDFLAGS += -fopenmp
endif

TARGET := clean_sdpa_bench
SRCS := main.cpp clean_sdpa.cpp
DEPS := clean_sdpa.h torch/extension.h Makefile ../../csrc/sdpa_flash2_neon_l3kv_impl.h

.PHONY: all clean run check

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ARCH_FLAGS) $(SRCS) -o $@ $(LDFLAGS)

run: $(TARGET)
	OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./$(TARGET) --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64 --iters=20 --warmup=5 --noncausal

check: $(TARGET)
	OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./$(TARGET) --B=1 --N=2 --L=16 --S=16 --E=16 --Ev=16 --iters=3 --warmup=1 --noncausal --check
	OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./$(TARGET) --B=1 --N=2 --L=16 --S=16 --E=16 --Ev=16 --iters=3 --warmup=1 --causal --check

clean:
	rm -f $(TARGET)
