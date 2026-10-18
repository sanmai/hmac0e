# Untracked per-machine overrides, e.g. `HOSTCXX = g++-15` when the local
# nvcc refuses the default host compiler.
-include Makefile.local

NVCC     ?= nvcc
CXX      ?= g++
HOSTCXX  ?= g++
ARCH     ?= sm_120  # RTX 5090; pass yours, e.g. make ARCH=sm_86

NVFLAGS  := -std=c++17 -O3 -arch=$(ARCH) -ccbin $(HOSTCXX) -Iinclude -Xptxas -O3
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude

HEADERS  := include/sha256.cuh include/hmac0e.cuh include/hmac_kernel.cuh
OBJS     := build/main.o build/hmac_kernel.o

.PHONY: all test host-test gpu-test check clean

all: hmac0e

build:
	@mkdir -p build

hmac0e: $(OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

build/%.o: src/%.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) -c $< -o $@

# CPU-only suite: SHA-256, HMAC, the predicate, and the hoisted candidate path.
# Runs with no GPU at all, so it works while the card is busy.
build/test_core: tests/test_core.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $< -o $@

test: build/test_core
	./build/test_core

# CLI parsing and persistence tests: CUDA toolkit to build, no GPU to run.
build/test_cli: tests/test_cli.cu src/main.cu build/hmac_kernel.o $(HEADERS) | build
	$(NVCC) $(NVFLAGS) -Xcompiler -Wall,-Wextra $< build/hmac_kernel.o -o $@

host-test: build/test_cli
	./build/test_cli

# GPU suite: kernel hit sets compared against the CPU oracle.
build/test_gpu: tests/test_gpu.cu build/hmac_kernel.o $(HEADERS) | build
	$(NVCC) $(NVFLAGS) $< build/hmac_kernel.o -o $@

gpu-test: build/test_gpu
	./build/test_gpu

check: test host-test gpu-test

clean:
	rm -rf build hmac0e
