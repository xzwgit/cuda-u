// cuda-u — command line front end: text and JSON reports.
// MIT License. See LICENSE.
#include "report.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char *kVersion = "0.0.2";

void usage() {
	printf(
		"cuda-u %s — CUDA device info & precision benchmark (MIT)\n"
		"\n"
		"usage: cuda-u [options]\n"
		"  -l, -list            list CUDA devices\n"
		"  -d N, -dev N         select device N (default: every device)\n"
		"  -j, -json [FILE]     emit JSON (to FILE, or stdout when omitted)\n"
		"  -n, -no-bench        skip benchmarks, static info only\n"
		"  -v, -version         print version\n"
		"  -h, -help            this help\n",
		kVersion);
}

std::string fmtRate(const Rate &r, bool integer) {
	if(!r.supported)
		return "N/A";
	if(r.opsPerSec <= 0)
		return "--";
	char buf[64];
	double g = r.opsPerSec / 1e9;
	if(g >= 1000.0)
		snprintf(buf, sizeof buf, "%.2f T%s/s", g / 1000.0, integer ? "iop" : "flop");
	else
		snprintf(buf, sizeof buf, "%.2f G%s/s", g, integer ? "iop" : "flop");
	return buf;
}

std::string fmtBytes(double bps) {
	char buf[64];
	double g = bps / (1000.0 * 1000.0 * 1000.0);
	snprintf(buf, sizeof buf, "%.2f GB/s", g);
	return buf;
}

const char *orNA(const std::string &s) {
	return s.empty() ? "N/A" : s.c_str();
}

std::string jsonEscape(const std::string &in) {
	std::string out;
	for(char c : in) {
		switch(c) {
		case '"':	out += "\\\""; break;
		case '\\':	out += "\\\\"; break;
		case '\n':	out += "\\n"; break;
		default:
			if((unsigned char)c < 0x20) {
				char b[8];
				snprintf(b, sizeof b, "\\u%04x", c);
				out += b;
			} else
				out += c;
		}
	}
	return out;
}

// one JSON number with 4 significant digits
std::string jnum(double v) {
	char buf[32];
	snprintf(buf, sizeof buf, "%.4g", v);
	return buf;
}

// dashed banners a la HiveOS: "---- Title ---..." padded to 78 columns
void banner(const char *title) {
	std::string line = "---- ";
	line += title;
	line += ' ';
	while(line.size() < 78)
		line += '-';
	printf("\n%s\n", line.c_str());
}

void rule() {
	printf("%s\n", std::string(78, '-').c_str());
}

void printText(const Report &r, bool staticOnly) {
	const DeviceFacts &f = r.dev;
	printf("cuda-u %s report - GPU %d: %s\n", kVersion, f.index, f.name.c_str());

	banner("Device");
	printf("  %-19s%s\n", "Name", f.name.c_str());
	// reference boards carry NVIDIA's own subsystem id -> N/A, raw id in brackets
	printf("  %-19s%s%s\n", "Board Vendor",
		f.boardVendor.empty() ? "N/A" : f.boardVendor.c_str(),
		f.boardVendorId.empty() ? "" : (" [" + f.boardVendorId + "]").c_str());
	printf("  %-19s%s\n", "Codename", orNA(f.codeName));
	printf("  %-19s%d.%d (%s)\n", "Compute", f.ccMajor, f.ccMinor, f.archName.c_str());
	printf("  %-19s%d / %d\n", "SMs / Cores", f.smCount, f.cores);
	if(f.tensorCores > 0)
		printf("  %-19s%d (4/SM)\n", "Tensor Cores", f.tensorCores);
	else
		printf("  %-19s%s\n", "Tensor Cores", "N/A");
	printf("  %-19s%d MHz (memory %d MHz)\n", "Clock", f.clockMHz, f.memClockMHz);
	if(f.powerLimitW > 0)
		printf("  %-19s%d W\n", "Power Limit", f.powerLimitW);
	else
		printf("  %-19s%s\n", "Power Limit", "N/A");
	if(f.pcieMaxGen > 0 && f.pcieMaxWidth > 0) {
		// capability first (whitepaper semantics); idle links downtrain, so
		// the live state only matters when it sits below capability
		printf("  %-19sGen %d x%d", "PCIe Link", f.pcieMaxGen, f.pcieMaxWidth);
		if(f.pcieGen > 0 && f.pcieWidth > 0 &&
		   (f.pcieGen != f.pcieMaxGen || f.pcieWidth != f.pcieMaxWidth))
			printf(" (now Gen %d x%d)", f.pcieGen, f.pcieWidth);
		printf("\n");
	} else {
		printf("  %-19s%s\n", "PCIe Link", "N/A");
	}
	printf("  %-19s%s / %s\n", "Regs/SMEM per SM",
		f.regsPerSm > 0 ? std::to_string(f.regsPerSm).c_str() : "N/A",
		f.l1PerSm > 0 ? (std::to_string(f.l1PerSm / 1024) + " KB").c_str() : "N/A");
	printf("  %-19s%s / CUDA %s\n", "Driver / Runtime",
		f.driverVersion.empty() ? "N/A" : f.driverVersion.c_str(),
		f.runtimeVersion.c_str());

	banner("Memory");
	printf("  %-19s%.0f MiB\n", "Total", f.memBytes / (1024.0 * 1024.0));
	printf("  %-19s%s / %s\n", "Type / Maker", orNA(f.memType), orNA(f.memMaker));
	printf("  %-19s%d bit\n", "Bus Width", f.busBits);
	if(f.memBandwidthSpec > 0)
		printf("  %-19s%.0f GB/s\n", "Bandwidth (spec)",
			f.memBandwidthSpec / 1e9);
	printf("  %-19s%.0f MiB (ECC %s)\n", "L2 Cache", f.l2Bytes / (1024.0 * 1024.0),
		f.ecc ? "on" : "off");

	if(!staticOnly) {
	banner("Bandwidth");
	printf("  %-19s%s\n", "H2D pinned", fmtBytes(r.h2dPinned).c_str());
		printf("  %-19s%s\n", "H2D pageable", fmtBytes(r.h2dPageable).c_str());
		printf("  %-19s%s\n", "D2H pinned", fmtBytes(r.d2hPinned).c_str());
		printf("  %-19s%s\n", "D2H pageable", fmtBytes(r.d2hPageable).c_str());
		printf("  %-19s%s\n", "D2D", fmtBytes(r.d2d).c_str());
	}

	if(!staticOnly) {
	banner("Compute");
	printf("  %-19s%16s%18s%18s\n", "", "vector path", "tensor (mma)", "GEMM (cuBLAS)");
	rule();
	printf("  %-19s%16s%18s\n", "FP64", fmtRate(r.fp64, false).c_str(), "N/A");
	printf("  %-19s%16s%18s\n", "FP32", fmtRate(r.fp32, false).c_str(), "N/A");
	printf("  %-19s%16s%18s\n", "TF32", "N/A", fmtRate(r.tf32t, false).c_str());
	printf("  %-19s%16s%18s%18s\n", "BF16", fmtRate(r.bf16, false).c_str(), fmtRate(r.bf16t, false).c_str(), fmtRate(r.bf16g, false).c_str());
	printf("  %-19s%16s%18s\n", "FP16", fmtRate(r.fp16, false).c_str(), fmtRate(r.fp16t, false).c_str());
	printf("  %-19s%16s%18s%18s\n", "FP8 (E4M3)", "N/A", fmtRate(r.fp8t, false).c_str(), fmtRate(r.fp8g, false).c_str());
	printf("  %-19s%16s%18s\n", "INT8 (DP4A)", fmtRate(r.int8dp4a, true).c_str(), "N/A");
	printf("  %-19s%16s%18s\n", "INT8", "N/A", fmtRate(r.int8t, true).c_str());
	printf("  %-19s%16s%18s\n", "FP4 (E2M1)", "N/A", fmtRate(r.fp4t, false).c_str());
	printf("  %-19s%16s%18s\n", "NVFP4", "N/A", fmtRate(r.fp4nv, false).c_str());
	printf("  %-19s%16s%18s\n", "INT4", "N/A", fmtRate(r.int4t, true).c_str());
	rule();
	}			// !staticOnly
}

std::string rateJson(const char *key, const Rate &r, bool integer) {
	std::string s = "    \"" + std::string(key) + "\": {\"value\": ";
	// null when the path is missing or the run skipped it (-n)
	s += (r.supported && r.opsPerSec > 0) ? jnum(r.opsPerSec / 1e9) : std::string("null");
	s += ", \"unit\": \"G";
	s += integer ? "op/s" : "flop/s";
	s += "\", \"supported\": ";
	s += r.supported ? "true" : "false";
	s += "},\n";
	return s;
}

std::string buildJson(const Report &r) {
	const DeviceFacts &f = r.dev;
	std::string j = "{\n";
	j += "  \"tool\": \"cuda-u\",\n  \"version\": \"" + std::string(kVersion) + "\",\n";
	j += "  \"device\": {\n";
	j += "    \"index\": " + std::to_string(f.index) + ",\n";
	j += "    \"name\": \"" + jsonEscape(f.name) + "\",\n";
	j += "    \"codename\": \"" + jsonEscape(f.codeName) + "\",\n";
	j += "    \"compute_capability\": \"" + std::to_string(f.ccMajor) + "." + std::to_string(f.ccMinor) + "\",\n";
	j += "    \"architecture\": \"" + jsonEscape(f.archName) + "\",\n";
	j += "    \"multiprocessors\": " + std::to_string(f.smCount) + ",\n";
	j += "    \"cuda_cores\": " + std::to_string(f.cores) + ",\n";
	j += "    \"tensor_cores\": " + std::to_string(f.tensorCores) + ",\n";
	j += "    \"regs_per_sm\": " + std::to_string(f.regsPerSm) + ",\n";
	j += "    \"l1_per_sm_bytes\": " + std::to_string(f.l1PerSm) + ",\n";
	j += "    \"power_limit_w\": " + std::to_string(f.powerLimitW) + ",\n";
	j += "    \"pcie_gen\": " + std::to_string(f.pcieGen) + ",\n";
	j += "    \"pcie_width\": " + std::to_string(f.pcieWidth) + ",\n";
	j += "    \"pcie_max_gen\": " + std::to_string(f.pcieMaxGen) + ",\n";
	j += "    \"pcie_max_width\": " + std::to_string(f.pcieMaxWidth) + ",\n";
	j += "    \"clock_mhz\": " + std::to_string(f.clockMHz) + ",\n";
	j += "    \"memory_clock_mhz\": " + std::to_string(f.memClockMHz) + ",\n";
	j += "    \"driver_version\": \"" + jsonEscape(f.driverVersion) + "\",\n";
	j += "    \"cuda_version\": \"" + jsonEscape(f.runtimeVersion) + "\"\n";
	j += "  },\n";
	j += "  \"board\": {\n";
	j += "    \"vendor\": \"" + jsonEscape(f.boardVendor) + "\",\n";
	j += "    \"vendor_id\": \"" + jsonEscape(f.boardVendorId) + "\"\n";
	j += "  },\n";
	j += "  \"memory\": {\n";
	j += "    \"total_mib\": " + jnum(f.memBytes / (1024.0 * 1024.0)) + ",\n";
	j += "    \"type\": \"" + jsonEscape(f.memType) + "\",\n";
	j += "    \"maker\": \"" + jsonEscape(f.memMaker) + "\",\n";
	j += "    \"bus_bits\": " + std::to_string(f.busBits) + ",\n";
	j += "    \"spec_bandwidth_gbps\": " + jnum(f.memBandwidthSpec / 1e9) + ",\n";
	j += "    \"l2_mib\": " + jnum(f.l2Bytes / (1024.0 * 1024.0)) + ",\n";
	j += "    \"ecc\": " + std::string(f.ecc ? "true" : "false") + "\n";
	j += "  },\n";
	j += "  \"bandwidth\": {\n";
	j += std::string("    \"h2d_pinned_gbps\": ")   + (r.h2dPinned   > 0 ? jnum(r.h2dPinned / 1e9)   : "null") + ",\n";
	j += std::string("    \"h2d_pageable_gbps\": ") + (r.h2dPageable > 0 ? jnum(r.h2dPageable / 1e9) : "null") + ",\n";
	j += std::string("    \"d2h_pinned_gbps\": ")   + (r.d2hPinned   > 0 ? jnum(r.d2hPinned / 1e9)   : "null") + ",\n";
	j += std::string("    \"d2h_pageable_gbps\": ") + (r.d2hPageable > 0 ? jnum(r.d2hPageable / 1e9) : "null") + ",\n";
	j += std::string("    \"d2d_gbps\": ")          + (r.d2d         > 0 ? jnum(r.d2d / 1e9)          : "null") + "\n";
	j += "  },\n";
	j += "  \"performance\": {\n";
	j += rateJson("fp64", r.fp64, false);
	j += rateJson("fp32", r.fp32, false);
	j += rateJson("tf32_tensor", r.tf32t, false);
	j += rateJson("bf16", r.bf16, false);
	j += rateJson("bf16_tensor", r.bf16t, false);
	j += rateJson("fp16", r.fp16, false);
	j += rateJson("fp16_tensor", r.fp16t, false);
	j += rateJson("fp8_tensor", r.fp8t, false);
	j += rateJson("int8_dp4a", r.int8dp4a, true);
	j += rateJson("int8_tensor", r.int8t, true);
	j += rateJson("fp4_tensor", r.fp4t, false);
	j += rateJson("bf16_gemm", r.bf16g, false);
	j += rateJson("fp8_gemm", r.fp8g, false);
	j += rateJson("nvfp4_tensor", r.fp4nv, false);
	j += rateJson("int4_tensor", r.int4t, true);
	j.pop_back();		// trailing comma of the last entry
	j.back() = '\n';
	j += "  }\n}\n";
	return j;
}

} // namespace

int main(int argc, char **argv) {
	bool list = false, json = false, bench = true, oneDevice = false;
	int device = 0;
	std::string jsonFile;

	for(int i = 1; i < argc; i++) {
		std::string a = argv[i];
		if(a == "-l" || a == "-list")
			list = true;
		else if(a == "-d" || a == "-dev") {
			if(++i >= argc || sscanf(argv[i], "%d", &device) != 1) {
				fprintf(stderr, "error: -dev needs a number\n");
				return 2;
			}
			oneDevice = true;
		} else if(a == "-j" || a == "-json") {
			json = true;
			if(i + 1 < argc && argv[i + 1][0] != '-')
				jsonFile = argv[++i];
		} else if(a == "-n" || a == "-no-bench")
			bench = false;
		else if(a == "-v" || a == "-version") {
			printf("cuda-u %s (MIT)\n", kVersion);
			return 0;
		} else if(a == "-h" || a == "-help") {
			usage();
			return 0;
		} else {
			fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
			usage();
			return 2;
		}
	}

	int count = 0;
	cudaError_t rc = cudaGetDeviceCount(&count);
	if(rc != cudaSuccess || count <= 0) {
		fprintf(stderr, "error: no CUDA devices (%s)\n", cudaGetErrorString(rc));
		return 1;
	}

	if(list) {
		printf("CUDA devices: %d\n", count);
		for(int i = 0; i < count; i++) {
			cudaDeviceProp p;
			if(cudaGetDeviceProperties(&p, i) == cudaSuccess)
				printf("  %d: %s (cc %d.%d, %d SMs)\n", i, p.name, p.major, p.minor,
					p.multiProcessorCount);
		}
		return 0;
	}
	if(device >= count) {
		fprintf(stderr, "error: device %d does not exist (0..%d)\n", device, count - 1);
		return 2;
	}

	// default: report every device; -d restricts to one
	std::vector<int> sel;
	if(oneDevice)
		sel.push_back(device);
	else
		for(int i = 0; i < count; i++)
			sel.push_back(i);

	int failed = 0;
	std::vector<std::string> docs;
	for(size_t k = 0; k < sel.size(); k++) {
		int i = sel[k];
		Report rep;
		std::string err;
		if(!collectFacts(i, rep.dev, err)) {
			fprintf(stderr, "error: device %d: %s\n", i, err.c_str());
			failed++;
			continue;
		}
		if(bench && !runSuite(i, rep, err)) {
			fprintf(stderr, "error: device %d benchmark failed: %s\n", i, err.c_str());
			failed++;
			continue;
		}
		if(json) {
			std::string doc = buildJson(rep);
			while(!doc.empty() && doc.back() == '\n')
				doc.pop_back();
			docs.push_back(doc);
		} else {
			if(k)
				printf("\n");
			printText(rep, !bench);
		}
	}

	if(json && !docs.empty()) {
		std::string doc;
		if(docs.size() == 1)
			doc = docs[0] + "\n";
		else {
			doc = "[\n";
			for(size_t k = 0; k < docs.size(); k++) {
				doc += docs[k];
				doc += k + 1 < docs.size() ? ",\n" : "\n";
			}
			doc += "]\n";
		}
		if(jsonFile.empty()) {
			fputs(doc.c_str(), stdout);
		} else {
			FILE *f = fopen(jsonFile.c_str(), "w");
			if(!f) {
				fprintf(stderr, "error: cannot write %s\n", jsonFile.c_str());
				return 1;
			}
			fputs(doc.c_str(), f);
			fclose(f);
		}
	}
	return failed ? 1 : 0;
}
