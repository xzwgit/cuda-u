// hidden_kernels.cu — SM120 隐藏格式测量内核（占位拼写，编译后由 patch_cubin 改位）
// E3M4: 用 e5m2 占位 (QMMA 0x27a/0x47a, A{78,82,83} B{79,84,85}: 001→010)
// E0M3: 用 e2m1 占位 (OMMA 0x47f, bit78/79: 1→0, 须 scale_vec::4X)
#include <cstdint>

#define KUNROLL 16

// FP8 E3M4 (占位: e5m2) — m16n8k32, 与公开 FP8 行同结构
extern "C" __global__ void kFp8E3M4t(float *out, int iters) {
	uint32_t a0 = 0x3c3c3c3cu, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for (int i = 0; i < iters; i++) {
#pragma unroll
		for (int j = 0; j < KUNROLL; j++)
			asm volatile("mma.sync.aligned.kind::f8f6f4.row.col.m16n8k32.f32.e5m2.e5m2.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}

// FP4 E0M3 (占位: e2m1, scale_vec::4X 强制) — m16n8k64 块缩放
extern "C" __global__ void kFp4E0M3t(float *out, int iters) {
	uint32_t a[4], b[2];
#pragma unroll
	for (int i = 0; i < 4; i++) a[i] = 0x11111111u * (i + 1);
	b[0] = 0x22222222u; b[1] = 0x33333333u;
	uint32_t sfa = 0x38383838u, sfb = 0x38383838u;   // ue4m3 = 1.0
	uint32_t lane = threadIdx.x & 31;
	float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
	for (int i = 0; i < iters; i++) {
#pragma unroll
		for (int j = 0; j < KUNROLL; j++)
			asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
				"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
				: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
				: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
				  "r"(b[0]), "r"(b[1]),
				  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)lane),
				  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)lane));
	}
	out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
}
