// cuda-u — NVFP4 GEMM benchmark entry (cuBLASLt, loaded lazily).
// MIT License. See LICENSE.
#pragma once

// Square NVFP4 (E2M1, block-scaled) GEMM throughput in TFLOPS, or 0 when
// libcublasLt is missing or the GPU has no NVFP4 algorithm.
double fp4GemmTflops(int size, int iters);
