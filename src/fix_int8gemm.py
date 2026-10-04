s = open("gemm_bench.cpp").read()

# INT8 用专用路径：INT8 输入 → INT32 输出（CUBLAS_COMPUTE_32I）
# BF16 输出不被支持，需改 layout 和输出缓冲

old_int8_case = """	case GEMM_INT8:
		abType = CUDA_R_8I;
		aBytes = elems;
		break;"""
new_int8_case = """	case GEMM_INT8:
		abType = CUDA_R_8I;
		aBytes = elems;
		break;
	// note: INT8 GEMM uses INT32 output, handled below"""

s = s.replace(old_int8_case, new_int8_case)

# 修 output 类型（INT8 → INT32）和输出缓冲大小
old_layout = """	rc |= (int)lib.layoutCreate(&lc, CUDA_R_16BF, size, size, size);
	rc |= (int)lib.layoutCreate(&ld, CUDA_R_16BF, size, size, size);"""
new_layout = """	cudaDataType_t outType = (gemmType == GEMM_INT8) ? CUDA_R_32I : CUDA_R_16BF;
	rc |= (int)lib.layoutCreate(&lc, outType, size, size, size);
	rc |= (int)lib.layoutCreate(&ld, outType, size, size, size);"""
if old_layout in s:
    s = s.replace(old_layout, new_layout)

# 修输出缓冲分配（INT8 需要 INT32 大小的 buffer）
old_malloc = "if(cudaMalloc(&a, aBytes) || cudaMalloc(&b, aBytes) ||\n\t   cudaMalloc(&d, elems * 2))"
new_malloc = "size_t dBytes = (gemmType == GEMM_INT8) ? elems * 4 : elems * 2;\n\tif(cudaMalloc(&a, aBytes) || cudaMalloc(&b, aBytes) ||\n\t   cudaMalloc(&d, dBytes))"
if old_malloc in s:
    s = s.replace(old_malloc, new_malloc)

# 修 cudaMemset for output
old_memset_d = "cudaMemset(d, 0, elems * 2);"
new_memset_d = "cudaMemset(d, 0, dBytes);"
if old_memset_d in s:
    s = s.replace(old_memset_d, new_memset_d)

open("gemm_bench.cpp", "w").write(s)
print("INT8 GEMM layout fixed (INT32 output)")
