# cuda-u — build. Requires CUDA toolkit 12.8+ (13.x tested).
NVCC ?= nvcc

GEN = -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_86,code=sm_86 \
      -gencode arch=compute_89,code=sm_89 \
      -gencode arch=compute_90,code=sm_90 \
      -gencode arch=compute_100a,code=sm_100a \
      -gencode arch=compute_103a,code=sm_103a \
      -gencode arch=compute_120a,code=sm_120a \
      -gencode arch=compute_90,code=compute_90

all: cuda-u

cuda-u: src/main.cpp src/gpu.cu src/report.h src/board.cpp src/board.h src/gemm_bench.cpp src/gemm_bench.h
	$(NVCC) -O3 -std=c++17 -t 0 --cudart=static $(GEN) \
		src/main.cpp src/gpu.cu src/board.cpp src/gemm_bench.cpp -o $@ -ldl -lcublasLt

clean:
	rm -f cuda-u

.PHONY: all clean
