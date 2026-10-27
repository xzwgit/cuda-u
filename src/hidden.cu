// hidden.cu — SM120/121 隐藏 Tensor Core 格式 (FP8 E3M4 / FP4 E0M3) 测量
//
// 这两种格式不存在于 PTX 类型系统, ptxas 拒绝其拼写。实现方式 (源自 Humming
// 项目的发现): 以 E5M2 / E2M1 占位拼写编译, 构建期对 cubin 做位级补丁
// (tools/patch_cubin.cpp), 运行时经 cuModuleLoadData 加载补丁镜像。
//
// SASS 位布局 (128-bit 指令):
//   QMMA 0x27a/0x47a: A 格式 {78,82,83} / B 格式 {79,84,85}: E5M2=001 → E3M4=010
//   OMMA 0x47f:       bit78(A) / bit79(B): E2M1=0 → E0M3=1; bit83=2X 形态对 E0M3 非法
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#include "report.h"

// tools/embed_cubins.py 生成的补丁镜像 (构建期产物, 全局作用域 extern)
extern const unsigned char cubin_e3m4_120[];
extern const size_t cubin_e3m4_120_size;
extern const unsigned char cubin_e0m3_120[];
extern const size_t cubin_e0m3_120_size;
extern const unsigned char cubin_e3m4_121[];
extern const size_t cubin_e3m4_121_size;
extern const unsigned char cubin_e0m3_121[];
extern const size_t cubin_e0m3_121_size;

namespace {

struct Blob {
	const unsigned char *data;
	size_t size;
};

int g_inited = -1;          // 0 = 镜像加载失败过, 1 = 成功
CUmodule g_mod = nullptr;
CUfunction g_e3m4 = nullptr, g_e0m3 = nullptr;

bool loadBlobs(const Blob &e3m4, const Blob &e0m3) {
	if (g_inited == 1) return true;
	if (g_inited == 0) return false;
	if (cuInit(0) != CUDA_SUCCESS) { g_inited = 0; return false; }
	if (cuModuleLoadData(&g_mod, e3m4.data) != CUDA_SUCCESS ||
	    cuModuleGetFunction(&g_e3m4, g_mod, "kFp8E3M4t") != CUDA_SUCCESS) { g_inited = 0; return false; }
	if (cuModuleLoadData(&g_mod, e0m3.data) != CUDA_SUCCESS ||
	    cuModuleGetFunction(&g_e0m3, g_mod, "kFp4E0M3t") != CUDA_SUCCESS) { g_inited = 0; return false; }
	g_inited = 1;
	return true;
}

double measureHidden(CUfunction fn, int blocks, int threads, double opsPerThreadPerIter) {
	float *out = nullptr;
	if (cudaMalloc(&out, (size_t)blocks * threads * sizeof(float)) != cudaSuccess)
		return 0;
	int iters = 200;
	void *args[] = { &out, &iters };
	cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, nullptr);
	cudaDeviceSynchronize();

	float ms = 0.f;
	for (int attempt = 0; attempt < 3; attempt++) {
		cudaEvent_t s, e;
		cudaEventCreate(&s);
		cudaEventCreate(&e);
		cudaEventRecord(s);
		cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, nullptr);
		cudaEventRecord(e);
		cudaEventSynchronize(e);
		cudaEventElapsedTime(&ms, s, e);
		cudaEventDestroy(s);
		cudaEventDestroy(e);
		if (ms >= 60.f)
			break;
		double scaled = (double)iters * (180.f / (ms > 0.1f ? ms : 0.1f));
		iters = scaled > 2000000 ? 2000000 : (int)scaled;
	}
	double best = 0;
	for (int rep = 0; rep < 3; rep++) {
		cudaEvent_t s, e;
		cudaEventCreate(&s);
		cudaEventCreate(&e);
		cudaEventRecord(s);
		cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, nullptr);
		cudaEventRecord(e);
		cudaEventSynchronize(e);
		cudaEventElapsedTime(&ms, s, e);
		cudaEventDestroy(s);
		cudaEventDestroy(e);
		double ops = (double)blocks * threads * opsPerThreadPerIter * iters;
		double rate = ops / (ms / 1e3);
		if (rate > best)
			best = rate;
	}
	cudaFree(out);
	return best;
}

} // namespace

bool runHiddenFormats(int device, Report &r, std::string &err) {
	DeviceFacts &f = r.dev;
	const int cc = f.ccMajor * 10 + f.ccMinor;
	// 镜像只为 120/121 编译; 别的架构保持 N/A
	if (cc != 120 && cc != 121) {
		r.fp8e3m4t.supported = false;
		r.fp4e0m3t.supported = false;
		return true;
	}
	Blob e3m4{cc == 121 ? cubin_e3m4_121 : cubin_e3m4_120, cc == 121 ? cubin_e3m4_121_size : cubin_e3m4_120_size};
	Blob e0m3{cc == 121 ? cubin_e0m3_121 : cubin_e0m3_120, cc == 121 ? cubin_e0m3_121_size : cubin_e0m3_120_size};
	if (!loadBlobs(e3m4, e0m3)) {
		err = "隐藏格式 cubin 加载失败";
		return false;
	}
	int blocks = f.smCount * 8, threads = 256;
	// kUnroll=16 (与内核一致); m16n8k32 = 8192 FLOP/warp-instr, m16n8k64 = 16384
	const double M_E3M4 = 16 * 8192.0 / 32;
	const double M_E0M3 = 16 * 16384.0 / 32;
	r.fp8e3m4t = {measureHidden(g_e3m4, blocks, threads, M_E3M4), true};
	r.fp4e0m3t = {measureHidden(g_e0m3, blocks, threads, M_E0M3), true};
	return true;
}
