ROOT := ../../..
FEATURES := ../features
I8GEMM := $(ROOT)/refs/i8gemm/lib

CXX ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++20 -pthread -Wall -Wextra -Werror \
	-march=armv8.6-a+sve+bf16+i8mm -I$(FEATURES) -I$(I8GEMM)
LDFLAGS += -pthread

TARGET := bench_elastic_nsplit
SOURCES := bench_elastic_nsplit.cpp $(FEATURES)/elastic_nsplit.cpp \
	$(ROOT)/csrc/moe_sve_fused_asm.S

.PHONY: all check clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(FEATURES)/elastic_nsplit.h $(I8GEMM)/gemm_params.h
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $@

check: $(TARGET)
	./$(TARGET) --stage w13 --m 120 --k 64 --n 64 --threads 4 \
		--low-threads 2 --epoch-rows 24 --copies 1 --check
	./$(TARGET) --stage w2 --m 120 --k 64 --n 64 --threads 4 \
		--low-threads 2 --epoch-rows 24 --copies 1 --check
	./$(TARGET) --stage w2_bf16 --m 120 --k 64 --n 64 --threads 4 \
		--low-threads 2 --epoch-rows 24 --copies 1 --check

clean:
	rm -f $(TARGET)
