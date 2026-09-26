// cuda-u — full-rate FP4 (NVFP4) throughput via a cuBLASLt block-scaled GEMM.
// The library is loaded with dlopen so the binary keeps no hard dependencies;
// when libcublasLt is absent (or the GPU predates NVFP4) the caller falls back
// to the mma.sync measurement.
// MIT License. See LICENSE.
#include "fp4gemm.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
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

// Returns TFLOPS of a square NVFP4 GEMM (TN, C/D in BF16, FP32 accumulate),
// or 0 when the library or an NVFP4 algorithm is unavailable.
double fp4GemmTflops(int size, int iters) {
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
	if(lib.descCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) { cleanup(); return 0; }

	cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
	const int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
	const size_t elems = (size_t)size * size;
	const size_t aBytes = elems / 2;				// 2 e2m1 per byte
	const size_t scaleBytes = elems / 16;				// one ue4m3 per 16 values

	if(cudaMalloc(&a, aBytes) || cudaMalloc(&b, aBytes) ||
	   cudaMalloc(&d, elems * 2) ||				// bf16 out
	   cudaMalloc(&sa, scaleBytes) || cudaMalloc(&sb, scaleBytes)) {
		cudaGetLastError(); cleanup(); return 0;
	}
	const size_t wsBytes = 32u << 20;
	cudaMalloc(&ws, wsBytes);
	cudaGetLastError();

	// every element 0.5 (e2m1 code 0b0010 -> byte 0x22), scales 1.0 (e4m3 0x38)
	cudaMemset(a, 0x22, aBytes);
	cudaMemset(b, 0x22, aBytes);
	cudaMemset(d, 0, elems * 2);
	cudaMemset(sa, 0x38, scaleBytes);
	cudaMemset(sb, 0x38, scaleBytes);

	if(lib.descSet(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) ||
	   lib.descSet(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) ||
	   lib.descSet(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof scaleMode) ||
	   lib.descSet(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof scaleMode) ||
	   lib.descSet(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof sa) ||
	   lib.descSet(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof sb) ||
	   lib.layoutCreate(&la, CUDA_R_4F_E2M1, size, size, size) ||
	   lib.layoutCreate(&lb, CUDA_R_4F_E2M1, size, size, size) ||
	   lib.layoutCreate(&lc, CUDA_R_16BF, size, size, size) ||
	   lib.layoutCreate(&ld, CUDA_R_16BF, size, size, size) ||
	   lib.prefCreate(&pref)) {
		cleanup(); return 0;
	}
	if(lib.prefSet(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof wsBytes)) {
		cleanup(); return 0;
	}

	cublasLtMatmulHeuristicResult_t heur[8];
	int found = 0;
	if(lib.heuristic(handle, op, la, lb, lc, ld, pref, 8, heur, &found) != CUBLAS_STATUS_SUCCESS || found <= 0) {
		cleanup(); return 0;					// GPU predates NVFP4
	}

	float alpha = 1.f, beta = 0.f;
	const double flopsPerCall = 2.0 * (double)size * size * size;
	cudaEventCreate(&ev0);
	cudaEventCreate(&ev1);

	for(int cand = 0; cand < found; cand++) {
		if(heur[cand].state != CUBLAS_STATUS_SUCCESS || heur[cand].workspaceSize > wsBytes)
			continue;
		// warm-up + correctness of the candidate
		if(lib.matmul(handle, op, &alpha, a, la, b, lb, &beta, d, lc, d, ld,
		              &heur[cand].algo, ws, wsBytes, 0) != CUBLAS_STATUS_SUCCESS) {
			cudaGetLastError();
			continue;
		}
		cudaDeviceSynchronize();

		for(int rep = 0; rep < 3; rep++) {
			cudaEventRecord(ev0);
			for(int i = 0; i < iters; i++)
				lib.matmul(handle, op, &alpha, a, la, b, lb, &beta, d, lc, d, ld,
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
