# cuda-u — build. Requires CUDA toolkit 12.8+ (13.x tested).
NVCC ?= nvcc
CXX ?= g++
PYTHON ?= python3

GEN = -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_86,code=sm_86 \
      -gencode arch=compute_89,code=sm_89 \
      -gencode arch=compute_90,code=sm_90 \
      -gencode arch=compute_100a,code=sm_100a \
      -gencode arch=compute_103a,code=sm_103a \
      -gencode arch=compute_120a,code=sm_120a \
      -gencode arch=compute_90,code=compute_90

# 隐藏格式 (FP8 E3M4 / FP4 E0M3, SM120/121): 占位拼写编译 → cubin 位级补丁 → 嵌入
HIDDEN = build/e3m4_120.cubin build/e0m3_120.cubin build/e3m4_121.cubin build/e0m3_121.cubin

all: cuda-u

build:
	mkdir -p build

build/patch_cubin: tools/patch_cubin.cpp | build
	$(CXX) -O2 -o $@ $<

build/hk_120.cubin: src/hidden_kernels.cu | build
	$(NVCC) -cubin -O3 -gencode arch=compute_120a,code=sm_120a $< -o $@

build/hk_121.cubin: src/hidden_kernels.cu | build
	$(NVCC) -cubin -O3 -gencode arch=compute_121a,code=sm_121a $< -o $@

build/e3m4_120.cubin: build/hk_120.cubin build/patch_cubin
	cp $< $@ && build/patch_cubin mma_e3m4_ab $@

build/e0m3_120.cubin: build/hk_120.cubin build/patch_cubin
	cp $< $@ && build/patch_cubin mma_e0m3_ab $@

build/e3m4_121.cubin: build/hk_121.cubin build/patch_cubin
	cp $< $@ && build/patch_cubin mma_e3m4_ab $@

build/e0m3_121.cubin: build/hk_121.cubin build/patch_cubin
	cp $< $@ && build/patch_cubin mma_e0m3_ab $@

src/hidden_cubins.cpp: tools/embed_cubins.py $(HIDDEN)
	$(PYTHON) $< $@ cubin_e3m4_120=build/e3m4_120.cubin cubin_e0m3_120=build/e0m3_120.cubin cubin_e3m4_121=build/e3m4_121.cubin cubin_e0m3_121=build/e0m3_121.cubin

cuda-u: src/main.cpp src/gpu.cu src/report.h src/board.cpp src/board.h src/gemm_bench.cpp src/gemm_bench.h src/hidden.cu src/hidden.h src/hidden_cubins.cpp
	$(NVCC) -O3 -std=c++17 -t 0 --cudart=static $(GEN) \
		src/main.cpp src/gpu.cu src/board.cpp src/gemm_bench.cpp src/hidden.cu src/hidden_cubins.cpp -o $@ -ldl -lcublasLt -lcuda

clean:
	rm -f cuda-u src/hidden_cubins.cpp
	rm -rf build

.PHONY: all clean
