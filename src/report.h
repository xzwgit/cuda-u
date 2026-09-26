// cuda-u — device facts and benchmark results.
// MIT License. See LICENSE.
#pragma once

#include <string>

struct DeviceFacts {
	int index = 0;
	std::string name;
	int ccMajor = 0, ccMinor = 0;
	std::string archName;
	std::string codeName;		// GA102 / GB202 ... from the name, empty if unknown
	int smCount = 0;
	int cores = 0;			// SMs x per-SM constant of the architecture
	int tensorCores = 0;		// SMs x 4 from Ampere on, 0 before Volta-era unknowns
	int regsPerSm = 0;		// 32-bit register file per SM
	int l1PerSm = 0;		// max shared/L1 per SM, bytes
	int clockMHz = 0, memClockMHz = 0;
	int busBits = 0;
	long long memBandwidthSpec = 0;	// busBits x 2 x memClock, bytes/s (GDDR)
	long long memBytes = 0;		// physical size (NVML), CUDA total as fallback
	long long l2Bytes = 0;
	bool ecc = false;
	int pciDomain = 0, pciBus = 0, pciDev = 0;
	int powerLimitW = 0;
	int pcieGen = 0, pcieWidth = 0, pcieMaxGen = 0, pcieMaxWidth = 0;
	std::string driverVersion;
	std::string runtimeVersion;
	// board / memory
	std::string boardVendor, boardVendorId, memMaker, memType;
};

// One throughput figure. Unsupported means the hardware lacks the path
// (report N/A); a zero value with supported=true means the measurement failed.
struct Rate {
	double opsPerSec = 0;
	bool supported = true;
};

struct Report {
	DeviceFacts dev;
	// bytes/s
	double h2dPinned = 0, h2dPageable = 0, d2hPinned = 0, d2hPageable = 0, d2d = 0;
	// ops/s, ordered from the widest float format down, tensor paths after
	// their vector counterparts
	Rate fp64, fp32, tf32t, bf16, bf16t, fp16, fp16t, fp8t, fp4t, fp4nv,
		int8dp4a, int8t, int4t;
};

// Collect static facts (fast, no kernels run).
bool collectFacts(int device, DeviceFacts &out, std::string &err);

// Run the full benchmark suite (fills the Rate fields and the copy rates).
bool runSuite(int device, Report &r, std::string &err);
