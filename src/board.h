// cuda-u — board vendor, memory vendor/type (NVAPI), physical memory size (NVML).
// MIT License. See LICENSE.
#pragma once

#include <string>

struct BoardInfo {
	std::string vendor;		// board partner, e.g. "ASUSTeK Computer Inc."
	std::string vendorId;		// raw PCI subsystem vendor, "0x1043"
	std::string memMaker;		// "Samsung", "Micron", ... empty if unknown
	std::string memType;		// "GDDR6X", "GDDR7", ... empty if unknown
	long long	memBytes = 0;	// physical memory size from NVML, 0 if unknown
	int		powerLimitW = 0;	// NVML default power limit, 0 if unknown
	int		pcieGen = 0, pcieWidth = 0;		// current link, 0 if unknown
	int		pcieMaxGen = 0, pcieMaxWidth = 0;	// device capability
};

// domain/bus/dev/func locate the card in sysfs; func is 0 for every GPU.
BoardInfo readBoardInfo(int domain, int bus, int dev, int func);
