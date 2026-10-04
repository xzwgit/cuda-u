// cuda-u — CUDA side: device facts + benchmark kernels.
// MIT License. See LICENSE.
#include <cstdint>          // uint32_t (CUDA 13.2 no longer pulls it in transitively)
#include "report.h"
#include "board.h"
#include "gemm_bench.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#include <cstdio>
#include <cstring>
#include <string>

// ------------------------------------------------------------------ facts
namespace {

int coresPerSm(int maj, int min) {
	switch(maj) {
	case 2:		return min >= 1 ? 48 : 32;		// Fermi
	case 3:		return 192;				// Kepler
	case 5:		return 128;				// Maxwell
	case 6:		return min == 0 ? 64 : 128;		// Pascal
	case 7:		return 64;				// Volta / Turing
	case 8:		return min == 0 ? 64 : 128;		// Ampere / Ada
	case 9:		return 128;				// Hopper
	case 10:	return 128;				// Blackwell (datacenter)
	case 12:	return 128;				// Blackwell (consumer)
	default:	return 0;
	}
}

const char *archName(int maj, int min) {
	switch(maj) {
	case 2:		return "Fermi";
	case 3:		return "Kepler";
	case 5:		return "Maxwell";
	case 6:		return "Pascal";
	case 7:		return min == 5 ? "Turing" : "Volta";
	case 8:		return min == 9 ? "Ada" : "Ampere";
	case 9:		return "Hopper";
	case 10:	return "Blackwell";
	case 12:	return "Blackwell";
	default:	return "";
	}
}

// chip codenames (NVIDIA whitepaper appendix tables), longest match first so
// "RTX 4070 Ti Super" (AD103) wins over "RTX 4070 Ti" (AD104)
struct { const char *name, *code; } kCodeNames[] = {
	// workstation parts first: "RTX 6000D" contains "RTX 6000", the plain
	// Turing fallback must stay last of this group
	{"RTX PRO 6000", "GB202"},	// Blackwell workstation/server edition
	{"RTX 6000D", "GB202"},	// China-market Blackwell workstation
	{"RTX 6000 Ada", "AD102"},
	{"RTX 6000", "TU102"},	// Turing-generation workstation
	{"RTX 5090", "GB202"},	{"RTX 5080", "GB203"},
	{"RTX 5070 Ti", "GB203"},	{"RTX 5070", "GB205"},
	{"RTX 4090", "AD102"},	{"RTX 4080", "AD103"},
	{"RTX 4070 Ti Super", "AD103"}, {"RTX 4070 Ti", "AD104"},
	{"RTX 4070", "AD104"},	{"RTX 4060", "AD107"},
	{"RTX 3090", "GA102"},	{"RTX 3080", "GA102"},
	{"RTX 3070 Ti", "GA104"},	{"RTX 3070", "GA104"},
	{"RTX 3060", "GA106"},	{"RTX 3050", "GA106"},
	{nullptr, nullptr},
};

const char *codeNameFor(const char *name) {
	if(!name || !*name)
		return nullptr;
	for(int i = 0; kCodeNames[i].name; i++)
		if(strstr(name, kCodeNames[i].name))
			return kCodeNames[i].code;
	return nullptr;
}

// The module line is either
//   ... NVIDIA UNIX x86_64 Kernel Module  580.105.08  Release Build ...
// or (open modules) "... Open Kernel Module for x86_64  580.105.08 ...",
// so take the first digit-led token after "Kernel Module".
std::string driverVersionFromProc() {
	FILE *f = fopen("/proc/driver/nvidia/version", "r");
	if(!f)
		return std::string();
	char line[512];
	std::string ver;
	while(fgets(line, sizeof line, f)) {
		const char *km = strstr(line, "Kernel Module");
		if(!km)
			continue;
		const char *p = km + strlen("Kernel Module");
		while(*p) {
			while(*p == ' ' || *p == '\t')
				p++;
			if(*p >= '0' && *p <= '9')
				break;
			while(*p && *p != ' ' && *p != '\t')
				p++;
		}
		if(*p >= '0' && *p <= '9') {
			char v[32];
			int i = 0;
			while(((*p >= '0' && *p <= '9') || *p == '.') && i < 30)
				v[i++] = *p++;
			v[i] = 0;
			ver = v;
		}
		break;
	}
	fclose(f);
	return ver;
}

} // namespace

bool collectFacts(int device, DeviceFacts &out, std::string &err) {
	cudaError_t rc;
	cudaDeviceProp p;
	if((rc = cudaGetDeviceProperties(&p, device)) != cudaSuccess) {
		err = cudaGetErrorString(rc);
		return false;
	}

	out.index = device;
	out.name = p.name;
	out.ccMajor = p.major;
	out.ccMinor = p.minor;
	out.archName = archName(p.major, p.minor);
	out.smCount = p.multiProcessorCount;
	out.cores = coresPerSm(p.major, p.minor) * p.multiProcessorCount;
	out.busBits = p.memoryBusWidth;
	out.l2Bytes = (long long)p.l2CacheSize;
	out.ecc = p.ECCEnabled != 0;
	out.pciDomain = p.pciDomainID;
	out.pciBus = p.pciBusID;
	out.pciDev = p.pciDeviceID;

	int v = 0;
	if(cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, device) == cudaSuccess)
		out.clockMHz = v / 1000;
	if(cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, device) == cudaSuccess)
		out.memClockMHz = v / 1000;
	if(cudaDeviceGetAttribute(&v, cudaDevAttrMaxRegistersPerMultiprocessor, device) == cudaSuccess)
		out.regsPerSm = v;
	if(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device) == cudaSuccess)
		out.l1PerSm = v;

	// 4 tensor cores per SM on every architecture since Ampere
	if(p.major >= 8)
		out.tensorCores = p.multiProcessorCount * 4;
	if(const char *code = codeNameFor(p.name))
		out.codeName = code;

	// GDDR theoretical peak: bus bits x 2 transfers/cycle x reported clock
	if(out.memClockMHz > 0 && out.busBits > 0)
		out.memBandwidthSpec = (long long)(out.busBits / 8.0 * 2.0 * out.memClockMHz * 1e6);

	out.driverVersion = driverVersionFromProc();
	int rt = 0;
	cudaRuntimeGetVersion(&rt);
	char buf[32];
	snprintf(buf, sizeof buf, "%d.%d", rt / 1000, (rt % 1000) / 10);
	out.runtimeVersion = buf;

	BoardInfo bi = readBoardInfo(p.pciDomainID, p.pciBusID, p.pciDeviceID, 0);
	out.boardVendor = bi.vendor;
	out.boardVendorId = bi.vendorId;
	out.memMaker = bi.memMaker;
	out.memType = bi.memType;
	out.memBytes = bi.memBytes > 0 ? bi.memBytes : (long long)p.totalGlobalMem;
	out.powerLimitW = bi.powerLimitW;
	out.pcieGen = bi.pcieGen;
	out.pcieWidth = bi.pcieWidth;
	out.pcieMaxGen = bi.pcieMaxGen;
	out.pcieMaxWidth = bi.pcieMaxWidth;
	return true;
}

// ------------------------------------------------------------- benchmarks
namespace {

constexpr int kUnroll = 256;		// scalar/tensor ops per outer iteration

// --------------------------------------------------------- vector kernels
__global__ void kFp64(float *out, int iters) {
	double a = threadIdx.x * 1e-9 + 1e-3, b = 1.0000000001;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 64")
		for(int j = 0; j < kUnroll; j++)
			a = fma(a, b, a);
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = (float)a;
}

__global__ void kFp32(float *out, int iters) {
	float a = threadIdx.x * 1e-6f + 1e-3f, b = 1.0000001f;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 64")
		for(int j = 0; j < kUnroll; j++)
			a = fmaf(a, b, a);
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
__global__ void kFp16v(float *out, int iters) {
	__half2 a = __float2half2_rn(0.25f + threadIdx.x * 1e-6f);
	__half2 b = __float2half2_rn(1.00390625f);
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 64")
		for(int j = 0; j < kUnroll; j++)
			a = __hfma2(a, b, a);
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = __half2float(__low2half(a));
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__global__ void kBf16v(float *out, int iters) {
	__nv_bfloat162 a = __float2bfloat162_rn(0.25f + threadIdx.x * 1e-6f);
	__nv_bfloat162 b = __float2bfloat162_rn(1.0078125f);
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 64")
		for(int j = 0; j < kUnroll; j++)
			a = __hfma2(a, b, a);
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = __bfloat162float(__low2bfloat16(a));
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
__global__ void kDp4a(float *out, int iters) {
	unsigned a = 0x01010101u * (threadIdx.x + 1), b = 0x02020202u;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 64")
		for(int j = 0; j < kUnroll; j++)
			a = __dp4a(a, b, a);
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = (float)a;
}
#endif

// --------------------------------------------------------- tensor kernels
// A warp-wide mma.sync carries 32 lanes; the launch-side accounting credits
// each lane ops/32 so blocks*threads*ops sums to the true rate.

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

__global__ void kTf32t(float *out, int iters) {
	uint32_t a0 = 0x3f000000u, a1 = 0x3e800000u, a2 = 0x3e000000u, a3 = 0x3d800000u;
	uint32_t b0 = 0x3f000000u, b1 = 0x3e800000u;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}

__global__ void kFp16t(float *out, int iters) {
	uint32_t a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}

__global__ void kBf16t(float *out, int iters) {
	uint32_t a0 = 0x3f803f80u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}

__global__ void kInt8t(float *out, int iters) {
	uint32_t a0 = 0x01010101u, a1 = 0x02020202u, a2 = 0x03030303u, a3 = 0x04040404u;
	uint32_t b0 = 0x01010101u, b1 = 0x02020202u;
	int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = (float)(d0 + d1 + d2 + d3);
}

#endif // __CUDA_ARCH__ >= 800

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
__global__ void kInt4t(float *out, int iters) {
	uint32_t a0 = 0x11111111u, a1 = 0x22222222u, b0 = 0x11111111u;
	int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
				: "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
				: "r"(a0), "r"(a1), "r"(b0));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = (float)(d0 + d1 + d2 + d3);
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
__global__ void kFp8t(float *out, int iters) {
	uint32_t a0 = 0x3c3c3c3cu, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}
#endif

// e2m1 exists only in the kind::f8f6f4 spelling and only for the consumer
// Blackwell family target, so plain targets must not even see the asm.
#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200)
__global__ void kFp4t(float *out, int iters) {
	uint32_t a0 = 0x11111111u, a1 = 0x22222222u, a2 = 0x33333333u, a3 = 0x44444444u;
	uint32_t b0 = 0x11111111u, b1 = 0x22222222u;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for(int i = 0; i < iters; i++) {
_Pragma("unroll 32")
		for(int j = 0; j < kUnroll; j++)
			asm volatile("mma.sync.aligned.kind::f8f6f4.row.col.m16n8k32.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}
#endif

// ------------------------------------------------------------ measurement
typedef void (*FloatKernel)(float *, int);

// Calibrate the iteration count for ~150 ms passes, then keep the best of 3.
double measure(FloatKernel fn, float *out, int blocks, int threads,
	       double opsPerThreadPerIter) {
	cudaEvent_t s, e;
	cudaEventCreate(&s);
	cudaEventCreate(&e);

	fn<<<blocks, threads>>>(out, 64);			// warm-up
	cudaDeviceSynchronize();

	int iters = 200;
	float ms = 0.f;
	for(int attempt = 0; attempt < 3; attempt++) {
		cudaEventRecord(s);
		fn<<<blocks, threads>>>(out, iters);
		cudaEventRecord(e);
		cudaEventSynchronize(e);
		cudaEventElapsedTime(&ms, s, e);
		if(ms >= 60.f)
			break;
		double scaled = (double)iters * (180.f / (ms > 0.1f ? ms : 0.1f));
		iters = scaled > 2000000 ? 2000000 : (int)scaled;
	}

	double best = 0;
	for(int rep = 0; rep < 3; rep++) {
		cudaEventRecord(s);
		fn<<<blocks, threads>>>(out, iters);
		cudaEventRecord(e);
		cudaEventSynchronize(e);
		cudaEventElapsedTime(&ms, s, e);
		double ops = (double)blocks * threads * opsPerThreadPerIter * iters;
		double rate = ops / (ms / 1e3);
		if(rate > best)
			best = rate;
	}
	cudaEventDestroy(s);
	cudaEventDestroy(e);
	return best;
}

} // namespace

bool runSuite(int device, Report &r, std::string &err) {
	cudaError_t rc;
	if((rc = cudaSetDevice(device)) != cudaSuccess) {
		err = cudaGetErrorString(rc);
		return false;
	}
	DeviceFacts &f = r.dev;
	int blocks = f.smCount * 8;
	int threads = 256;
	const int cc = f.ccMajor * 10 + f.ccMinor;

	float *scratch = nullptr;
	if((rc = cudaMalloc(&scratch, (size_t)blocks * threads * sizeof(float))) != cudaSuccess) {
		err = cudaGetErrorString(rc);
		return false;
	}

	// per-lane ops per outer iteration; mma values are warp ops divided by 32
	const double SCALAR2 = kUnroll * 2.0;			// fma chains
	const double PACKED4 = kUnroll * 4.0;			// half2/bf162 chains
	const double DP4A = kUnroll * 8.0;			// 4 mul + 4 add
	const double M_TF32 = kUnroll * (2048.0 / 32);
	const double M_16B = kUnroll * (4096.0 / 32);
	const double M_32B = kUnroll * (8192.0 / 32);

	r.fp64 = {measure(kFp64, scratch, blocks, threads, SCALAR2), true};
	r.fp32 = {measure(kFp32, scratch, blocks, threads, SCALAR2), true};

	/* Launch sites mirror the kernel guards: the device pass of every target
	 * architecture parses this function too, so a kernel excluded for that
	 * arch must not be referenced unguarded. Runtime CC gates pick what runs. */
	r.fp16.supported = cc >= 53;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
	if(r.fp16.supported)
		r.fp16.opsPerSec = measure(kFp16v, scratch, blocks, threads, PACKED4);
#endif

	r.bf16.supported = cc >= 80;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
	if(r.bf16.supported)
		r.bf16.opsPerSec = measure(kBf16v, scratch, blocks, threads, PACKED4);
#endif

	r.int8dp4a.supported = cc >= 61;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
	if(r.int8dp4a.supported)
		r.int8dp4a.opsPerSec = measure(kDp4a, scratch, blocks, threads, DP4A);
#endif

	// tf32 / bf16 / fp16 / int8 mma paths, all sm_80+
	if(cc >= 80) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
		r.tf32t = {measure(kTf32t, scratch, blocks, threads, M_TF32), true};
		r.bf16t = {measure(kBf16t, scratch, blocks, threads, M_16B), true};
		r.fp16t = {measure(kFp16t, scratch, blocks, threads, M_16B), true};
		r.int8t = {measure(kInt8t, scratch, blocks, threads, M_32B), true};
#endif
	} else {
		r.tf32t.supported = r.bf16t.supported = r.fp16t.supported = r.int8t.supported = false;
	}

	r.fp8t.supported = cc >= 89;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
	if(r.fp8t.supported)
		r.fp8t.opsPerSec = measure(kFp8t, scratch, blocks, threads, M_32B);
#endif

	r.int4t.supported = cc >= 75;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
	if(r.int4t.supported)
		r.int4t.opsPerSec = measure(kInt4t, scratch, blocks, threads, M_32B);
#endif

	r.fp4t.supported = cc >= 120;
#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200)
	if(r.fp4t.supported) {
		/* mma.sync only exposes m16n8k32 e2m1, which shares the FP8 issue
		 * slot; the full-rate block-scaled format is the NVFP4 row below. */
		r.fp4t.opsPerSec = measure(kFp4t, scratch, blocks, threads, M_32B);
	}
#endif

	// GEMM paths via cuBLASLt (tcgen05.mma on sm_100+, full library-achievable
	// rate). mma.sync above only exercises the legacy warp-level path, which on
	// datacenter Blackwell is ~4x below the tcgen05 rate.
	r.bf16g.supported = cc >= 80;
	if(r.bf16g.supported) {
		double g = gemmBenchTflops(GEMM_BF16, 8192, 8);
		r.bf16g.supported = g > 0;
		if(r.bf16g.supported)
			r.bf16g.opsPerSec = g * 1e12;
	}

	r.fp8g.supported = cc >= 89;
	if(r.fp8g.supported) {
		double g = gemmBenchTflops(GEMM_FP8, 8192, 8);
		r.fp8g.supported = g > 0;
		if(r.fp8g.supported)
			r.fp8g.opsPerSec = g * 1e12;
	}

	r.fp4nv.supported = cc >= 100;
	if(r.fp4nv.supported) {
		double gemm = gemmBenchTflops(GEMM_NVFP4, 8192, 8);
		r.fp4nv.supported = gemm > 0;
		if(r.fp4nv.supported)
			r.fp4nv.opsPerSec = gemm * 1e12;
	}

	// ------------------------------------------------------ copy bandwidth
	const size_t sz = 64ull << 20;
	char *hostPage = (char *)malloc(sz);
	char *hostPin = nullptr;
	char *devA = nullptr, *devB = nullptr;
	cudaHostAlloc(&hostPin, sz, cudaHostAllocDefault);
	cudaMalloc(&devA, sz);
	cudaMalloc(&devB, sz);

	cudaEvent_t s, e;
	cudaEventCreate(&s);
	cudaEventCreate(&e);
	auto copyRate = [&](int kind) -> double {
		double best = 0;
		for(int rep = 0; rep < 5; rep++) {
			cudaEventRecord(s);
			switch(kind) {
			case 0: cudaMemcpy(devA, hostPin, sz, cudaMemcpyHostToDevice); break;
			case 1: cudaMemcpy(devA, hostPage, sz, cudaMemcpyHostToDevice); break;
			case 2: cudaMemcpy(hostPin, devA, sz, cudaMemcpyDeviceToHost); break;
			case 3: cudaMemcpy(hostPage, devA, sz, cudaMemcpyDeviceToHost); break;
			default: cudaMemcpy(devB, devA, sz, cudaMemcpyDeviceToDevice); break;
			}
			cudaEventRecord(e);
			cudaEventSynchronize(e);
			float ms;
			cudaEventElapsedTime(&ms, s, e);
			double bytes = (kind == 4) ? 2.0 * (double)sz : (double)sz;
			double rate = bytes / (ms / 1e3);
			if(rate > best)
				best = rate;
		}
		return best;
	};
	r.h2dPinned = copyRate(0);
	r.h2dPageable = copyRate(1);
	r.d2hPinned = copyRate(2);
	r.d2hPageable = copyRate(3);
	r.d2d = copyRate(4);

	free(hostPage);
	cudaFreeHost(hostPin);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(scratch);
	return true;
}
