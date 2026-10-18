# Untracked per-machine overrides, e.g. `HOSTCXX = g++-15` when the local
# nvcc refuses the default host compiler.
-include Makefile.local

NVCC     ?= nvcc
CXX      ?= g++
HOSTCXX  ?= g++
# sm_120 is the RTX 5090. Pass yours, e.g. make ARCH=sm_86. Keep the comment
# off this line: make keeps the whitespace before a trailing comment, so
# "ARCH ?= sm_120  # ..." would define an arch with spaces in it.
ARCH     ?= sm_120

NVFLAGS  := -std=c++17 -O3 -arch=$(ARCH) -ccbin $(HOSTCXX) -Iinclude -Xptxas -O3
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude

HEADERS  := include/sha256.cuh include/hmac0e.cuh include/hmac_kernel.cuh \
            include/launch_range.cuh

# CUDA output depends on ARCH and the host compiler, so its objects and
# executables live in a per-arch directory: changing ARCH builds fresh ones
# instead of silently relinking the previous arch, and changing back reuses
# them. Editing the flags below rebuilds them in place.
CUDA_BUILD := build/$(ARCH)
OBJS       := $(CUDA_BUILD)/main.o $(CUDA_BUILD)/hmac_kernel.o
BUILD_DEPS := Makefile $(wildcard Makefile.local)

.PHONY: all hmac0e test host-test gpu-test check clean

all: hmac0e

# test_core needs no CUDA toolkit, so it stays outside the arch directory.
build:
	@mkdir -p build

$(CUDA_BUILD):
	@mkdir -p $(CUDA_BUILD)

$(CUDA_BUILD)/hmac0e: $(OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

# Phony, so ./hmac0e always follows the selected ARCH: a timestamp rule would
# keep a newer binary built for another arch. The rename replaces the file
# atomically, so a running search keeps its loaded executable.
hmac0e: $(CUDA_BUILD)/hmac0e
	@cmp -s $< $@ || { cp $< $@.tmp && mv -f $@.tmp $@ && echo "installed $< as $@"; }

$(CUDA_BUILD)/%.o: src/%.cu $(HEADERS) $(BUILD_DEPS) | $(CUDA_BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# CPU-only suite: SHA-256, HMAC, the predicate, and the hoisted candidate path.
# Runs with no GPU at all, so it works while the card is busy.
build/test_core: tests/test_core.cpp $(HEADERS) $(BUILD_DEPS) | build
	$(CXX) $(CXXFLAGS) $< -o $@

test: build/test_core
	./build/test_core

# CLI parsing and persistence tests: CUDA toolkit to build, no GPU to run.
$(CUDA_BUILD)/test_cli: tests/test_cli.cu src/main.cu $(CUDA_BUILD)/hmac_kernel.o \
                        $(HEADERS) $(BUILD_DEPS) | $(CUDA_BUILD)
	$(NVCC) $(NVFLAGS) -Xcompiler -Wall,-Wextra $< $(CUDA_BUILD)/hmac_kernel.o -o $@

host-test: $(CUDA_BUILD)/test_cli
	./$(CUDA_BUILD)/test_cli

# GPU suite: kernel hit sets compared against the CPU oracle.
$(CUDA_BUILD)/test_gpu: tests/test_gpu.cu $(CUDA_BUILD)/hmac_kernel.o \
                        $(HEADERS) $(BUILD_DEPS) | $(CUDA_BUILD)
	$(NVCC) $(NVFLAGS) $< $(CUDA_BUILD)/hmac_kernel.o -o $@

gpu-test: $(CUDA_BUILD)/test_gpu
	./$(CUDA_BUILD)/test_gpu

check: test host-test gpu-test

clean:
	rm -rf build hmac0e hmac0e.tmp
