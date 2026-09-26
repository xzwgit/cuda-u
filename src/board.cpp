// cuda-u — board vendor (PCI subsystem), memory vendor/type (NVAPI),
// physical memory size (NVML). Linux, unprivileged.
// MIT License. See LICENSE.
#include "board.h"

#include <ctype.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#if defined(__linux__)

#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------- strings
const char *kMakerNames[] = {
	"", "Samsung", "Qimonda", "Elpida", "Etron", "Nanya", "Hynix",
	"Mosel", "Winbond", "ESMT", "Micron",
};

const char *memMakerName(unsigned v) {
	return (v > 0 && v < sizeof(kMakerNames) / sizeof(kMakerNames[0])) ? kMakerNames[v] : nullptr;
}

const char *memTypeName(unsigned v) {
	switch(v) {
	case 8:		return "GDDR5";
	case 14:	return "GDDR6";
	case 15:	return "GDDR6X";
	case 16:	return "GDDR7";
	default:	return nullptr;
	}
}

// small partner table used when the system has no pci.ids installed
struct { unsigned id; const char *name; } kPartners[] = {
	{0x1043, "ASUSTeK Computer Inc."}, {0x1458, "Gigabyte Technology"},
	{0x1462, "Micro-Star MSI"},	 {0x196e, "PNY"},	{0x19da, "ZOTAC"},
	{0x3842, "EVGA"},		 {0x1b4c, "GALAX"},	{0x1569, "Palit"},
	{0x10b0, "Gainward"},		 {0x7377, "Colorful"},	{0x1acc, "Inno3D"},
	{0x1849, "ASRock"},		 {0x1b0a, "Pegatron"},	{0x1028, "Dell"},
	{0x103c, "HP"},			 {0x17aa, "Lenovo"},
};

const char *kPciIdsPaths[] = {
	"/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids",
	"/usr/share/pci.ids", nullptr,
};

bool readFileTrim(const char *path, char *buf, size_t cap) {
	int fd = open(path, O_RDONLY);
	if(fd < 0)
		return false;
	ssize_t n = read(fd, buf, cap - 1);
	close(fd);
	if(n <= 0)
		return false;
	buf[n] = 0;
	while(n > 0 && isspace((unsigned char)buf[n - 1]))
		buf[--n] = 0;
	return true;
}

// Vendor names from the system pci.ids ("1043  ASUSTeK Computer Inc.").
std::string vendorFromPciIds(unsigned id) {
	for(int i = 0; kPciIdsPaths[i]; i++) {
		FILE *f = fopen(kPciIdsPaths[i], "r");
		if(!f)
			continue;
		char line[512];
		while(fgets(line, sizeof line, f)) {
			if(line[0] == '#' || line[0] == '\n' || line[0] == '\t' || line[0] == ' ')
				continue;
			unsigned vid;
			if(sscanf(line, "%x", &vid) != 1 || vid != id)
				continue;
			fclose(f);
			char *p = strstr(line, "  ");
			if(!p)
				return std::string();
			p += 2;
			std::string name(p);
			while(!name.empty() && (name.back() == '\n' || name.back() == '\r'))
				name.pop_back();
			return name;
		}
		fclose(f);
	}
	return std::string();
}

// ------------------------------------------------------------- NVAPI side
// Query-interface ids for the entry points this tool needs; resolved through
// the driver's own nvapi bridge library, no extra packages required.
constexpr unsigned kNvapiInitialize = 0x0150E828;
constexpr unsigned kNvapiEnumGPUs = 0xE5AC921F;
constexpr unsigned kNvapiGetRamMaker = 0x42AEA16A;
constexpr unsigned kNvapiGetRamType = 0x57F7CAAC;
constexpr unsigned kNvapiGetBusId = 0x1BE0B8E5;

typedef void *(*QueryInterfaceFn)(unsigned);

struct NvapiData {
	unsigned maker = 0, type = 0;
	bool haveMaker = false, haveType = false;
	bool matched = false;
};

NvapiData nvapiRamInfo(int bus) {
	NvapiData r;
	void *lib = dlopen("libnvidia-api.so.1", RTLD_NOW | RTLD_GLOBAL);
	if(!lib)
		lib = dlopen("libnvidia-api.so", RTLD_NOW | RTLD_GLOBAL);
	if(!lib)
		return r;

	QueryInterfaceFn q = (QueryInterfaceFn)dlsym(lib, "nvapi_QueryInterface");
	if(!q)
		return r;

	auto init = (int (*)(void))q(kNvapiInitialize);
	auto enumGpus = (int (*)(void **, int *))q(kNvapiEnumGPUs);
	auto getMaker = (int (*)(void *, void *))q(kNvapiGetRamMaker);
	auto getType = (int (*)(void *, void *))q(kNvapiGetRamType);
	auto getBus = (int (*)(void *, void *))q(kNvapiGetBusId);
	if(!init || !enumGpus || init() != 0)
		return r;

	void *handles[32];
	int count = 0;
	if(enumGpus(handles, &count) != 0 || count <= 0)
		return r;
	if(count > 32)
		count = 32;

	int idx = -1;
	for(int i = 0; i < count; i++) {			// match on PCI bus number
		unsigned char buf[64] = {0};
		if(getBus && getBus(handles[i], buf) == 0 && *(unsigned *)buf == (unsigned)bus) {
			idx = i;
			break;
		}
	}
	if(idx < 0 && bus < count)				// fall back to enumeration order
		idx = bus;
	if(idx < 0)
		return r;

	unsigned char buf[64] = {0};
	if(getMaker && getMaker(handles[idx], buf) == 0) {
		r.maker = *(unsigned *)buf;
		r.haveMaker = true;
	}
	memset(buf, 0, sizeof buf);
	if(getType && getType(handles[idx], buf) == 0) {
		r.type = *(unsigned *)buf;
		r.haveType = true;
	}
	r.matched = true;
	return r;
}

// -------------------------------------------------------------- NVML side
long long nvmlTotalBytes(const char *pciBusId) {
	void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
	if(!lib)
		return 0;
	auto init = (int (*)(void))dlsym(lib, "nvmlInit_v2");
	auto byBus = (int (*)(const char *, void **))dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
	auto getMem = (int (*)(void *, void *))dlsym(lib, "nvmlDeviceGetMemoryInfo");
	if(!init || !byBus || !getMem || init() != 0)
		return 0;
	void *h = nullptr;
	if(byBus(pciBusId, &h) != 0)
		return 0;
	struct { unsigned long long total, used, free_; } mi = {0, 0, 0};
	if(getMem(h, &mi) != 0)
		return 0;
	return (long long)mi.total;
}

// default power management limit in watts (the card's TGP), 0 if unavailable
int nvmlPowerLimitW(const char *pciBusId) {
	void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
	if(!lib)
		return 0;
	auto init = (int (*)(void))dlsym(lib, "nvmlInit_v2");
	auto byBus = (int (*)(const char *, void **))dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
	auto getPower = (int (*)(void *, unsigned *))dlsym(lib, "nvmlDeviceGetPowerManagementLimit");
	if(!init || !byBus || !getPower || init() != 0)
		return 0;
	void *h = nullptr;
	unsigned mw = 0;
	if(byBus(pciBusId, &h) != 0 || getPower(h, &mw) != 0)
		return 0;
	return (int)(mw / 1000);
}

// sysfs link speed is "2.5/5/8/16/32/64 [GT/s]" (sometimes "16.0"); map to gen
int pcieGenFromSpeed(const char *text) {
	if(!text || !*text)
		return 0;
	double gts = atof(text);
	if(gts <= 0)
		return 0;
	if(gts < 3)	return 1;
	if(gts < 6)	return 2;
	if(gts < 9)	return 3;
	if(gts < 17)	return 4;
	if(gts < 33)	return 5;
	return 6;
}

int readIntFile(const char *path) {
	char buf[32];
	if(!readFileTrim(path, buf, sizeof buf))
		return 0;
	return atoi(buf);
}

} // namespace

BoardInfo readBoardInfo(int domain, int bus, int dev, int func) {
	BoardInfo b;
	char path[256], text[128];

	// board partner from the PCI subsystem vendor id
	snprintf(path, sizeof path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/subsystem_vendor",
		domain, bus, dev, func);
	if(readFileTrim(path, text, sizeof text)) {
		unsigned id = (unsigned)strtoul(text, nullptr, 0);
		char idbuf[16];
		snprintf(idbuf, sizeof idbuf, "0x%04x", id);
		b.vendorId = idbuf;
		if(id != 0x10de) {				// NVIDIA's own id = reference board
			std::string name = vendorFromPciIds(id);
			if(name.empty())
				for(auto &p : kPartners)
					if(p.id == id) {
						name = p.name;
						break;
					}
			b.vendor = name;			// stays empty -> caller shows N/A
		}
	}

	// memory vendor/type through the driver's nvapi bridge
	NvapiData n = nvapiRamInfo(bus);
	if(n.haveMaker && memMakerName(n.maker))
		b.memMaker = memMakerName(n.maker);
	if(n.haveType && memTypeName(n.type))
		b.memType = memTypeName(n.type);

	// physical memory size (what nvidia-smi prints, not the CUDA usable part)
	// and default power limit, both through NVML
	char busId[24];
	snprintf(busId, sizeof busId, "%04x:%02x:%02x.%x", domain, bus, dev, func);
	b.memBytes = nvmlTotalBytes(busId);
	b.powerLimitW = nvmlPowerLimitW(busId);

	// PCIe link state from sysfs: current vs. device capability
	snprintf(path, sizeof path,
		"/sys/bus/pci/devices/%04x:%02x:%02x.%x/current_link_speed",
		domain, bus, dev, func);
	if(readFileTrim(path, text, sizeof text))
		b.pcieGen = pcieGenFromSpeed(text);
	b.pcieWidth = readIntFile(
		(snprintf(path, sizeof path,
			  "/sys/bus/pci/devices/%04x:%02x:%02x.%x/current_link_width",
			  domain, bus, dev, func), path));
	snprintf(path, sizeof path,
		"/sys/bus/pci/devices/%04x:%02x:%02x.%x/max_link_speed",
		domain, bus, dev, func);
	if(readFileTrim(path, text, sizeof text))
		b.pcieMaxGen = pcieGenFromSpeed(text);
	b.pcieMaxWidth = readIntFile(
		(snprintf(path, sizeof path,
			  "/sys/bus/pci/devices/%04x:%02x:%02x.%x/max_link_width",
			  domain, bus, dev, func), path));
	return b;
}

#else // !__linux__

BoardInfo readBoardInfo(int, int, int, int) {
	return BoardInfo();
}

#endif
