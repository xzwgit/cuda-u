// cuda-u — data passed between modules.
// MIT License. See LICENSE.
#ifndef CUDA_U_REPORT_H
#define CUDA_U_REPORT_H

#include <string>

struct Rate {
	double opsPerSec = 0;
	bool supported = false;
};

struct DeviceFacts {
	int index = 0;
	std::string name;
	int ccMajor = 0, ccMinor = 0;
	std::string archName;
	int smCount = 0, cores = 0;
	int busBits = 0;
	long long l2Bytes = 0;
	bool ecc = false;
	int clockMHz = 0, memClockMHz = 0;
	int regsPerSm = 0, l1PerSm = 0;
	int tensorCores = 0;
	long long memBandwidthSpec = 0;
	long long memBytes = 0;
	std::string driverVersion, runtimeVersion;
	std::string codeName;
	int pciDomain = 0, pciBus = 0, pciDev = 0;
	// board info (from sysfs)
	std::string boardVendor, boardVendorId;
	std::string memMaker, memType;
	int powerLimitW = 0;
	int pcieGen = 0, pcieWidth = 0;
	int pcieMaxGen = 0, pcieMaxWidth = 0;
};

struct Report {
	DeviceFacts dev;

	// bandwidth (bytes/s)
	double h2dPinned = 0, h2dPageable = 0;
	double d2hPinned = 0, d2hPageable = 0;
	double d2d = 0;

	// vector paths
	Rate fp64, fp32, fp16, bf16, int8dp4a;

	// tensor paths — mma.sync (kernel-level)
	Rate tf32t, fp16t, bf16t, fp8t, int8t, int4t, fp4t;

	// GEMM paths — cuBLASLt (library-achievable; uses tcgen05 on sm_100+)
	Rate bf16g, fp8g, int8g, fp4nv;
};

bool collectFacts(int device, DeviceFacts &out, std::string &err);
bool runSuite(int device, Report &r, std::string &err);

#endif
