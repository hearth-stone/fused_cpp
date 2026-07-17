ROOT := ../../..
FEATURES := ../features
I8GEMM := $(ROOT)/refs/i8gemm/lib

CXX ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++20 -pthread -Wall -Wextra -Werror \
	-march=armv8.6-a+sve+bf16+i8mm -I$(FEATURES) -I$(I8GEMM) -I$(ROOT)/csrc
LDFLAGS += -pthread

TARGETS := bench_elastic_nsplit bench_unfused_pipeline bench_single_core_weight_window \
	bench_fragmented_route_pipeline bench_m12_llc_pollution bench_route_merge_tree
ELASTIC_SOURCES := bench_elastic_nsplit.cpp $(FEATURES)/elastic_nsplit.cpp \
	$(ROOT)/csrc/moe/arm/sve_bf16/kernels.S
UNFUSED_SOURCES := bench_unfused_pipeline.cpp $(FEATURES)/unfused_pipeline.cpp \
	$(ROOT)/csrc/moe/arm/sve_bf16/kernels.S
SINGLE_CORE_WINDOW_SOURCES := bench_single_core_weight_window.cpp \
	$(ROOT)/csrc/moe/arm/sve_bf16/kernels.S
M12_LLC_POLLUTION_SOURCES := bench_m12_llc_pollution.cpp \
	$(ROOT)/csrc/moe/arm/sve_bf16/kernels.S
FRAGMENTED_ROUTE_SOURCES := bench_fragmented_route_pipeline.cpp $(FEATURES)/unfused_pipeline.cpp \
	$(ROOT)/csrc/moe/arm/sve_bf16/kernels.S
ROUTE_MERGE_SOURCES := bench_route_merge_tree.cpp $(ROOT)/csrc/moe/arm/sve_bf16/route_merge.cpp

.PHONY: all check clean

all: $(TARGETS)

bench_elastic_nsplit: $(ELASTIC_SOURCES) $(FEATURES)/elastic_nsplit.h $(I8GEMM)/gemm_params.h
	$(CXX) $(CXXFLAGS) $(ELASTIC_SOURCES) $(LDFLAGS) -o $@

bench_unfused_pipeline: $(UNFUSED_SOURCES) $(FEATURES)/unfused_pipeline.h $(I8GEMM)/gemm_params.h
	$(CXX) $(CXXFLAGS) $(UNFUSED_SOURCES) $(LDFLAGS) -o $@

bench_single_core_weight_window: $(SINGLE_CORE_WINDOW_SOURCES) $(I8GEMM)/gemm_params.h
	$(CXX) $(CXXFLAGS) $(SINGLE_CORE_WINDOW_SOURCES) $(LDFLAGS) -o $@

bench_m12_llc_pollution: $(M12_LLC_POLLUTION_SOURCES) $(I8GEMM)/gemm_params.h
	$(CXX) $(CXXFLAGS) $(M12_LLC_POLLUTION_SOURCES) $(LDFLAGS) -o $@

bench_fragmented_route_pipeline: $(FRAGMENTED_ROUTE_SOURCES) $(FEATURES)/unfused_pipeline.h \
	$(I8GEMM)/gemm_params.h
	$(CXX) $(CXXFLAGS) $(FRAGMENTED_ROUTE_SOURCES) $(LDFLAGS) -o $@

bench_route_merge_tree: $(ROUTE_MERGE_SOURCES) $(ROOT)/csrc/moe/common/route_merge.h
	$(CXX) $(CXXFLAGS) $(ROUTE_MERGE_SOURCES) $(LDFLAGS) -o $@

check: $(TARGETS)
	./bench_elastic_nsplit --stage w13 --m 120 --k 64 --n 64 --threads 4 \
		--low-threads 2 --epoch-rows 24 --copies 1 --check
	./bench_elastic_nsplit --stage w2 --m 120 --k 64 --n 64 --threads 4 \
		--low-threads 2 --epoch-rows 24 --copies 1 --check
	./bench_elastic_nsplit --stage w2_bf16 --m 120 --k 64 --n 64 --threads 4 \
		--low-threads 2 --epoch-rows 24 --copies 1 --check
	./bench_unfused_pipeline --experts 2 --m 24 --h 64 --f 64 \
		--threads-per-expert 2 --w13-ranges 2 --copies 1 --check
	./bench_single_core_weight_window --m 24 --k 64 --n 64 \
		--warmup 1 --runs 2 --cold-tail-mib 1 --check
	./bench_m12_llc_pollution --victim-mib 1 --polluter-experts 1 \
		--polluter-panels 1 --trials 1 --evict-mib 1 --cpu 0
	./bench_fragmented_route_pipeline --teams 2 --base-routes 24 \
		--replaced-teams 1 --split-factor 2 --hidden 64 --intermediate 64 \
		--threads-per-team 2 --schedule dynamic --w13-ranges 2 --copies 1 --check
	for top_k in 1 2 3 4 5 6 7 8 9; do \
		./bench_route_merge_tree --tokens 5 --top-k $$top_k --hidden 103 --threads 4 \
			--warmup 1 --runs 2 --source both --check || exit 1; \
	done

clean:
	rm -f $(TARGETS)
