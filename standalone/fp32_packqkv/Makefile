CXX ?= c++
ARCH_FLAGS ?= -march=armv8.6-a+sve+bf16+i8mm
CXXFLAGS ?= -pthread -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -Wno-class-memaccess -fPIC -std=c++17
CPPFLAGS ?= -I. -Icsrc -Icsrc/sdpa_microkernels
LDFLAGS ?=
USE_OPENMP ?= 1

ifeq ($(USE_OPENMP),1)
CXXFLAGS += -DFUSED_CPP_HAS_OMP=1 -fopenmp
LDFLAGS += -fopenmp
endif

TARGET := fp32_packqkv_sdpa_bench
LIB := libfused_cpp_sdpa_fp32_packqkv.so
SRCS := main.cpp fp32_packqkv_sdpa.cpp
LIB_SRCS := fp32_packqkv_sdpa.cpp
DEPS := fp32_packqkv_sdpa.h Makefile fp32_packqkv_sdpa.version \
        csrc/sdpa_common.h \
        csrc/sdpa_standalone_shim.h \
        csrc/sdpa_tile_sizes.h \
        csrc/sdpa_pack_utils.h \
        csrc/sdpa_profile.h \
        csrc/sdpa_flash2_neon_l3kv_impl.h \
        csrc/sdpa_microkernels/neon_cache_config.h \
        csrc/sdpa_microkernels/neon_cache_microkernels.h \
        csrc/sdpa_microkernels/mk_traits.h \
        csrc/sdpa_microkernels/impls/mk_qk_packqk_seq4_bmajor_pv_pquad.h

.PHONY: all clean run check lib

all: $(TARGET) $(LIB)

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ARCH_FLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(LIB): $(LIB_SRCS) $(DEPS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ARCH_FLAGS) -shared $(LIB_SRCS) -o $@ $(LDFLAGS) -Wl,--version-script=fp32_packqkv_sdpa.version

lib: $(LIB)

run: $(TARGET)
	OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./$(TARGET) --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64 --iters=20 --warmup=5 --noncausal

check: $(TARGET)
	OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./$(TARGET) --B=1 --N=2 --L=16 --S=16 --E=16 --Ev=16 --iters=3 --warmup=1 --noncausal --check
	OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE ./$(TARGET) --B=1 --N=2 --L=16 --S=16 --E=16 --Ev=16 --iters=3 --warmup=1 --causal --check

clean:
	rm -f $(TARGET) $(LIB)
