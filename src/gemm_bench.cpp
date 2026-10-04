// cuda-u — full-rate GEMM throughput via cuBLASLt (BF16 / FP8 / NVFP4).
// The library is loaded with dlopen so the binary keeps no hard dependencies;
// when libcublasLt is absent (or the GPU predates the format) the caller falls
// back to the mma.sync measurement.
// MIT License. See LICENSE.
#include "gemm_bench.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// stable C ABI entry points, resolved at run time
typedef cublasStatus_t (*Create_t)(cublasLtHandle_t *);
typedef cublasStatus_t (*Destroy_t)(cublasLtHandle_t);
typedef cublasStatus_t (*DescCreate_t)(cublasLtMatmulDesc_t *, cublasComputeType_t, cudaDataType_t);
typedef cublasStatus_t (*DescSet_t)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *, size_t);
typedef cublasStatus_t (*DescDestroy_t)(cublasLtMatmulDesc_t);
typedef cublasStatus_t (*LayoutCreate_t)(cublasLtMatrixLayout_t *, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef cublasStatus_t (*LayoutDestroy_t)(cublasLtMatrixLayout_t);
typedef cublasStatus_t (*PrefCreate_t)(cublasLtMatmulPreference_t *);
typedef cublasStatus_t (*PrefSet_t)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void *, size_t);
typedef cublasStatus_t (*PrefDestroy_t)(cublasLtMatmulPreference_t);
typedef cublasStatus_t (*Heuristic_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, const cublasLtMatrixLayout_t,
                                      const cublasLtMatrixLayout_t, const cublasLtMatrixLayout_t,
                                      const cublasLtMatrixLayout_t, const cublasLtMatmulPreference_t,
                                      int, cublasLtMatmulHeuristicResult_t *, int *);
typedef cublasStatus_t (*Matmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *,
                                   const cublasLtMatrixLayout_t, const void *, const cublasLtMatrixLayout_t,
                                   const void *, const void *, const cublasLtMatrixLayout_t, void *,
                                   const cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *,
                                   void *, size_t, cudaStream_t);

struct Lib {
	void *h = nullptr;
	Create_t create = nullptr;
	Destroy_t destroy = nullptr;
	DescCreate_t descCreate = nullptr;
	DescSet_t descSet = nullptr;
	DescDestroy_t descDestroy = nullptr;
	LayoutCreate_t layoutCreate = nullptr;
	LayoutDestroy_t layoutDestroy = nullptr;
	PrefCreate_t prefCreate = nullptr;
	PrefSet_t prefSet = nullptr;
	PrefDestroy_t prefDestroy = nullptr;
	Heuristic_t heuristic = nullptr;
	Matmul_t matmul = nullptr;

	bool load() {
		if(h)
			return matmul != nullptr;
		for(const char *name : {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"})
			if((h = dlopen(name, RTLD_NOW | RTLD_GLOBAL)) != nullptr)
				break;
		if(!h)
			return false;
		create = (Create_t)dlsym(h, "cublasLtCreate");
		destroy = (Destroy_t)dlsym(h, "cublasLtDestroy");
		descCreate = (DescCreate_t)dlsym(h, "cublasLtMatmulDescCreate");
		descSet = (DescSet_t)dlsym(h, "cublasLtMatmulDescSetAttribute");
		descDestroy = (DescDestroy_t)dlsym(h, "cublasLtMatmulDescDestroy");
		layoutCreate = (LayoutCreate_t)dlsym(h, "cublasLtMatrixLayoutCreate");
		layoutDestroy = (LayoutDestroy_t)dlsym(h, "cublasLtMatrixLayoutDestroy");
		prefCreate = (PrefCreate_t)dlsym(h, "cublasLtMatmulPreferenceCreate");
		prefSet = (PrefSet_t)dlsym(h, "cublasLtMatmulPreferenceSetAttribute");
		prefDestroy = (PrefDestroy_t)dlsym(h, "cublasLtMatmulPreferenceDestroy");
		heuristic = (Heuristic_t)dlsym(h, "cublasLtMatmulAlgoGetHeuristic");
		matmul = (Matmul_t)dlsym(h, "cublasLtMatmul");
		return create && descCreate && descSet && layoutCreate && prefCreate &&
		       heuristic && matmul;
	}
};

} // namespace

// Returns TFLOPS of a square GEMM (TN, FP32 accumulate), or 0 when the
// library or an algorithm for the requested format is unavailable.
//
// On datacenter Blackwell (sm_100+), mma.sync only exercises the legacy
// warp-level tensor core path; cuBLASLt dispatches tcgen05.mma which
// achieves the full datasheet rate. This measurement answers "what can a
// real GEMM workload achieve" rather than "what does one mma.sync
// instruction issue".
double gemmBenchTflops(int gemmType, int size, int iters) {
	Lib lib;
	if(!lib.load())
		return 0;

	cublasLtHandle_t handle = nullptr;
	cublasLtMatmulDesc_t op = nullptr;
	cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
	cublasLtMatmulPreference_t pref = nullptr;
	void *a = nullptr, *b = nullptr, *d = nullptr, *sa = nullptr, *sb = nullptr, *ws = nullptr;
	cudaEvent_t ev0 = nullptr, ev1 = nullptr;
	double best = 0;

	auto cleanup = [&]() {
		if(ev0) cudaEventDestroy(ev0);
		if(ev1) cudaEventDestroy(ev1);
		if(pref) lib.prefDestroy(pref);
		if(la) lib.layoutDestroy(la);
		if(lb) lib.layoutDestroy(lb);
		if(lc) lib.layoutDestroy(lc);
		if(ld) lib.layoutDestroy(ld);
		if(op) lib.descDestroy(op);
		if(handle) lib.destroy(handle);
		cudaFree(ws); cudaFree(sa); cudaFree(sb);
		cudaFree(a); cudaFree(b); cudaFree(d);
	};

	if(lib.create(&handle) != CUBLAS_STATUS_SUCCESS) { cleanup(); return 0; }
	cublasComputeType_t compType = (gemmType == GEMM_INT8) ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
	cudaDataType_t scaleType = (gemmType == GEMM_INT8) ? CUDA_R_32I : CUDA_R_32F;
	if(lib.descCreate(&op, compType, scaleType) != CUBLAS_STATUS_SUCCESS) { cleanup(); return 0; }

	cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
	const size_t elems = (size_t)size * size;

	// format selection
	cudaDataType_t abType;
	size_t aBytes, scaleBytes = 0;
	bool useScales = false;
	switch(gemmType) {
	case GEMM_BF16:
		abType = CUDA_R_16BF;
		aBytes = elems * 2;
		break;
	case GEMM_FP8:
		abType = CUDA_R_8F_E4M3;
		aBytes = elems;
		break;
	case GEMM_NVFP4: {
		abType = CUDA_R_4F_E2M1;
		aBytes = elems / 2;
		scaleBytes = elems / 16;
		useScales = true;
		break;
	}
	case GEMM_INT8:
		abType = CUDA_R_8I;
		aBytes = elems;
		break;
	// note: INT8 GEMM uses INT32 output, handled below
	default:
		cleanup();
		return 0;
	}
	const int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

	size_t dBytes = (gemmType == GEMM_INT8) ? elems * 4 : elems * 2;
	if(cudaMalloc(&a, aBytes) || cudaMalloc(&b, aBytes) ||
	   cudaMalloc(&d, dBytes)) {  // bf16 out for all formats
		cudaGetLastError(); cleanup(); return 0;
	}
	if(useScales && (cudaMalloc(&sa, scaleBytes) || cudaMalloc(&sb, scaleBytes))) {
		cudaGetLastError(); cleanup(); return 0;
	}
	const size_t wsBytes = 32u << 20;
	cudaMalloc(&ws, wsBytes);
	cudaGetLastError();

	// init: values that produce well-defined results without overflow
	switch(gemmType) {
	case GEMM_BF16:
		// 0.5 in bf16 = 0x3F00
		cudaMemset(a, 0x00, aBytes);  // bf16 0 = all-zero bytes
		// fill with 0.5 pattern (2 bytes per element, little-endian 0x3F00)
		{
			unsigned short v = 0x3F00;
			cudaMemset(a, 0, aBytes);
			// just fill with 0.5 pattern byte-by-byte won't work for bf16,
			// so use 0 which is safe for all formats
		}
		break;
	case GEMM_FP8:
		// 0.5 in e4m3 = 0x30
		cudaMemset(a, 0x30, aBytes);
		cudaMemset(b, 0x30, aBytes);
		break;
	case GEMM_INT8:
		// value 1 in int8 = 0x01
		cudaMemset(a, 0x01, aBytes);
		cudaMemset(b, 0x01, aBytes);
		break;
	case GEMM_NVFP4:
		// 0.5 in e2m1 = 0b0010 -> byte 0x22 (two per byte)
		cudaMemset(a, 0x22, aBytes);
		cudaMemset(b, 0x22, aBytes);
		if(sa) cudaMemset(sa, 0x38, scaleBytes);  // e4m3 1.0 = 0x38
		if(sb) cudaMemset(sb, 0x38, scaleBytes);
		break;
	}
	if(gemmType != GEMM_FP8 && gemmType != GEMM_NVFP4) {
		// BF16: fill with 0.5 (bf16 0x3F00, little-endian bytes 0x00 0x3F)
		cudaMemset(a, 0x00, aBytes);  // start with zeros
		// For throughput benchmark, content doesn't matter much — use memset
		// to avoid overhead of proper initialization
		cudaMemset(a, 0x3F, aBytes);
		cudaMemset(b, 0x3F, aBytes);
	}
	cudaMemset(d, 0, dBytes);

	int rc = 0;
	rc |= (int)lib.descSet(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
	rc |= (int)lib.descSet(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
	if(useScales) {
		rc |= (int)lib.descSet(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof scaleMode);
		rc |= (int)lib.descSet(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof scaleMode);
		rc |= (int)lib.descSet(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof sa);
		rc |= (int)lib.descSet(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof sb);
	}
	rc |= (int)lib.layoutCreate(&la, abType, size, size, size);
	rc |= (int)lib.layoutCreate(&lb, abType, size, size, size);
	cudaDataType_t outType = (gemmType == GEMM_INT8) ? CUDA_R_32I : CUDA_R_16BF;
	rc |= (int)lib.layoutCreate(&lc, outType, size, size, size);
	rc |= (int)lib.layoutCreate(&ld, outType, size, size, size);
	rc |= (int)lib.prefCreate(&pref);
	if(rc != 0) { cleanup(); return 0; }

	if(lib.prefSet(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof wsBytes)) {
		cleanup(); return 0;
	}

	cublasLtMatmulHeuristicResult_t heur[8];
	int found = 0;
	if(lib.heuristic(handle, op, la, lb, lc, ld, pref, 8, heur, &found) != CUBLAS_STATUS_SUCCESS || found <= 0) {
		cleanup(); return 0;  // GPU predates this format
	}

		float alpha_f = 1.f, beta_f = 0.f;
	int alpha_i = 1, beta_i = 0;
	void* alpha = (gemmType == GEMM_INT8) ? (void*)&alpha_i : (void*)&alpha_f;
	void* beta = (gemmType == GEMM_INT8) ? (void*)&beta_i : (void*)&beta_f;
	const double flopsPerCall = 2.0 * (double)size * size * size;
	cudaEventCreate(&ev0);
	cudaEventCreate(&ev1);

	for(int cand = 0; cand < found; cand++) {
		if(heur[cand].state != CUBLAS_STATUS_SUCCESS || heur[cand].workspaceSize > wsBytes)
			continue;
		if(lib.matmul(handle, op, alpha, a, la, b, lb, beta, d, lc, d, ld,
		              &heur[cand].algo, ws, wsBytes, 0) != CUBLAS_STATUS_SUCCESS) {
			cudaGetLastError();
			continue;
		}
		cudaDeviceSynchronize();

		for(int rep = 0; rep < 3; rep++) {
			cudaEventRecord(ev0);
			for(int i = 0; i < iters; i++)
				lib.matmul(handle, op, alpha, a, la, b, lb, beta, d, lc, d, ld,
				           &heur[cand].algo, ws, wsBytes, 0);
			cudaEventRecord(ev1);
			cudaEventSynchronize(ev1);
			float ms;
			cudaEventElapsedTime(&ms, ev0, ev1);
			double tf = flopsPerCall * iters / (ms / 1e3) / 1e12;
			if(tf > best)
				best = tf;
		}
	}

	cleanup();
	return best;
}
