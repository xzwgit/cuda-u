// cuda-u — GEMM benchmark via cuBLASLt (BF16 / FP8 / NVFP4).
// MIT License. See LICENSE.
#ifndef CUDA_U_GEMM_BENCH_H
#define CUDA_U_GEMM_BENCH_H

enum {
	GEMM_BF16 = 0,
	GEMM_FP8  = 1,
	GEMM_NVFP4 = 2,
	GEMM_INT8  = 3,
};

// Returns TFLOPS of a square GEMM (TN, FP32 accumulate), or 0 on failure.
double gemmBenchTflops(int gemmType, int size, int iters);

#endif
