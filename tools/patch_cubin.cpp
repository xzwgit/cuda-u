// patch_cubin.cpp — SM120 隐藏格式 cubin 补丁器 (cuda-u 用, 位语义源自 Humming 的发现)
// 用法: patch_cubin <mode> <file.cubin>   mode: mma_e3m4_ab | mma_e0m3_ab
// QMMA 0x27a/0x47a: A{78,82,83}/B{79,84,85} E5M2=001 → E3M4=010
// OMMA 0x47f: bit78(A)/bit79(B) E2M1=0 → E0M3=1 (mma.sync 极性, 与 tcgen05 相反); bit83=2X 非法拒绝
// 仅接受 SM120/121 的 cubin
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int getbit(const std::vector<uint8_t> &d, size_t o, int n) {
	return (d[o + n / 8] >> (n % 8)) & 1;
}
static void setbit(std::vector<uint8_t> &d, size_t o, int n, int v) {
	if (v) d[o + n / 8] |= 1u << (n % 8);
	else d[o + n / 8] &= ~(1u << (n % 8));
}

int main(int argc, char **argv) {
	if (argc != 3) { fprintf(stderr, "用法: %s <mma_e3m4_ab|mma_e0m3_ab> <cubin>\n", argv[0]); return 2; }
	bool is_e3m4 = !strcmp(argv[1], "mma_e3m4_ab");
	if (!is_e3m4 && strcmp(argv[1], "mma_e0m3_ab")) { fprintf(stderr, "未知 mode\n"); return 2; }

	std::ifstream f(argv[2], std::ios::binary);
	std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (d.size() < 64 || memcmp(d.data(), "\x7f" "ELF", 4) || d[4] != 2) { fprintf(stderr, "非 ELF64\n"); return 2; }
	uint16_t emachine;
	uint32_t eflags;
	memcpy(&emachine, d.data() + 0x12, 2);
	memcpy(&eflags, d.data() + 0x30, 4);
	if (emachine != 0xBE) { fprintf(stderr, "非 CUDA cubin\n"); return 3; }
	uint32_t sm = eflags & 0xff;
	if (sm < 75) sm = (eflags >> 8) & 0xff;
	if (sm != 120 && sm != 121) { fprintf(stderr, "目标 SM%u 非 120/121\n", sm); return 3; }

	uint64_t e_shoff;
	uint16_t e_shents, e_shnum, e_shstr;
	memcpy(&e_shoff, d.data() + 0x28, 8);
	memcpy(&e_shents, d.data() + 0x3A, 2);
	memcpy(&e_shnum, d.data() + 0x3C, 2);
	memcpy(&e_shstr, d.data() + 0x3E, 2);
	uint64_t stroff;
	memcpy(&stroff, d.data() + e_shstr * e_shents + 0x18, 8);

	int matched = 0, patched = 0;
	for (int s = 0; s < e_shnum; s++) {
		const uint8_t *sh = d.data() + e_shoff + s * e_shents;
		uint32_t sh_type;
		uint64_t sh_flag, sh_off, sh_size;
		memcpy(&sh_type, sh + 0x04, 4);
		memcpy(&sh_flag, sh + 0x08, 8);
		memcpy(&sh_off, sh + 0x18, 8);
		memcpy(&sh_size, sh + 0x20, 8);
		if (sh_type != 1 || !(sh_flag & 0x4)) continue;   // PROGBITS + EXECINSTR
		for (uint64_t o = sh_off; o + 16 <= sh_off + sh_size; o += 16) {
			uint32_t op = d[o] | ((d[o + 1] & 0x0f) << 8);
			if (is_e3m4) {
				if (op != 0x47a && op != 0x27a) continue;
				int a78 = getbit(d, o, 78), a82 = getbit(d, o, 82), a83 = getbit(d, o, 83);
				int b79 = getbit(d, o, 79), b84 = getbit(d, o, 84), b85 = getbit(d, o, 85);
				matched++;
				if (a78 == 1 && a82 == 0 && a83 == 0 && b79 == 1 && b84 == 0 && b85 == 0) {
					setbit(d, o, 78, 0); setbit(d, o, 82, 1);
					setbit(d, o, 79, 0); setbit(d, o, 84, 1);
					patched++;
				}
			} else {
				if (op != 0x47f) continue;
				if (getbit(d, o, 83)) { fprintf(stderr, "OMMA 为 2X 形态, E0M3 非法, 拒绝\n"); return 5; }
				matched++;
				if (getbit(d, o, 78) == 0 && getbit(d, o, 79) == 0) {
					setbit(d, o, 78, 1); setbit(d, o, 79, 1);
					patched++;
				}
			}
		}
	}
	if (!matched) { fprintf(stderr, "未找到目标指令\n"); return 5; }
	std::ofstream w(argv[2], std::ios::binary);
	w.write((const char *)d.data(), d.size());
	printf("%s: 匹配 %d 条指令, 补丁 %d 条 → %s\n", argv[1], matched, patched, argv[2]);
	return patched ? 0 : 5;
}
